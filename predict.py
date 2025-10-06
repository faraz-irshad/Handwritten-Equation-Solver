import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from models import EfficientCNN, ImprovedCNN, load_model, get_transforms
from segment import segment_image
import json

class Solver:
    def __init__(self, digit_model_path='models/digit_cnn.pth', op_model_path='models/op_cnn.pth', device='cpu'):
        self.device = device
        
        # Load multiple model types for ensemble
        self.models = {'digit': [], 'op': []}
        
        # Try to load EfficientCNN
        try:
            self.models['digit'].append(load_model(EfficientCNN, digit_model_path, device, 10))
            self.models['op'].append(load_model(EfficientCNN, op_model_path, device, 4))
            print("Loaded EfficientCNN models")
        except:
            pass
        
        # Try to load ImprovedCNN as backup/ensemble
        try:
            self.models['digit'].append(load_model(ImprovedCNN, digit_model_path.replace('.pth', '_improved.pth'), device, 10))
            self.models['op'].append(load_model(ImprovedCNN, op_model_path.replace('.pth', '_improved.pth'), device, 4))
            print("Loaded ImprovedCNN models")
        except:
            pass
        
        # Fallback to simple CNN
        if not self.models['digit']:
            try:
                from models import SimpleCNN
                self.models['digit'].append(load_model(SimpleCNN, digit_model_path, device, 10))
                self.models['op'].append(load_model(SimpleCNN, op_model_path, device, 4))
                print("Loaded SimpleCNN models")
            except:
                raise Exception("No models could be loaded")
        
        self.transform = get_transforms(train=False)
        
        # Load operator classes
        try:
            with open('models/op_classes.json', 'r') as f:
                self.op_classes = json.load(f)
        except:
            self.op_classes = ['+', '-', '×', '÷']
    
    def preprocess_symbol(self, img_arr):
        pil = Image.fromarray(img_arr).convert('L')
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        return tensor
    
    def predict_ensemble_tta(self, tensor, models):
        """Ensemble + TTA for maximum accuracy"""
        all_predictions = []
        
        for model in models:
            model_preds = []
            
            # Original
            with torch.no_grad():
                pred = F.softmax(model(tensor), dim=1)
                model_preds.append(pred)
            
            # Augmentations
            augmentations = [
                torch.flip(tensor, [3]),  # Horizontal flip
                torch.rot90(tensor, 1, [2, 3]),  # 90° rotation
                torch.rot90(tensor, -1, [2, 3]),  # -90° rotation
            ]
            
            for aug_tensor in augmentations:
                with torch.no_grad():
                    pred = F.softmax(model(aug_tensor), dim=1)
                    model_preds.append(pred)
            
            # Average this model's predictions
            model_avg = torch.mean(torch.stack(model_preds), dim=0)
            all_predictions.append(model_avg)
        
        # Ensemble average
        return torch.mean(torch.stack(all_predictions), dim=0)
    
    def context_aware_prediction(self, symbols_probs, position):
        """Use context to improve predictions"""
        digit_probs, op_probs = symbols_probs
        
        # Context rules
        if position == 0:  # First symbol must be digit
            return 'digit', digit_probs
        elif position % 2 == 0:  # Even positions should be digits
            # Boost digit confidence if it's reasonable
            if digit_probs.max() > 0.3:
                boosted = digit_probs * 1.5
                return 'digit', boosted / boosted.sum()
        else:  # Odd positions should be operators
            # Boost operator confidence if it's reasonable
            if op_probs.max() > 0.3:
                boosted = op_probs * 1.5
                return 'op', boosted / boosted.sum()
        
        # Default: use higher confidence
        if digit_probs.max() > op_probs.max():
            return 'digit', digit_probs
        else:
            return 'op', op_probs
    
    def predict_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        
        symbols = segment_image(image)
        if not symbols:
            return "No symbols detected", None
        
        expression = ''
        confidences = []
        
        for i, symbol in enumerate(symbols):
            tensor = self.preprocess_symbol(symbol)
            
            # Get ensemble predictions with TTA
            digit_probs = self.predict_ensemble_tta(tensor, self.models['digit'])[0]
            op_probs = self.predict_ensemble_tta(tensor, self.models['op'])[0]
            
            # Apply context-aware prediction
            pred_type, final_probs = self.context_aware_prediction((digit_probs, op_probs), i)
            
            confidence = final_probs.max().item()
            confidences.append(confidence)
            
            if pred_type == 'digit':
                symbol_pred = str(final_probs.argmax().item())
            else:
                symbol_pred = self.op_classes[final_probs.argmax()]
            
            expression += symbol_pred
        
        # Post-process with grammar correction
        expression = self._grammar_correction(expression)
        
        # Validate expression structure
        if not self._is_valid_expression(expression):
            return f"Invalid expression: {expression}", None
        
        # Evaluate expression
        try:
            python_expr = expression.replace('×', '*').replace('÷', '/')
            result = eval(python_expr)
            avg_confidence = np.mean(confidences)
            return f"{expression} (conf: {avg_confidence:.2f})", result
        except Exception as e:
            return f"Error evaluating {expression}: {str(e)}", None
    
    def _grammar_correction(self, expr):
        """Fix common OCR errors using grammar rules"""
        if not expr:
            return expr
        
        operators = ['+', '-', '×', '÷']
        corrected = list(expr)
        
        # Fix alternating pattern violations
        for i in range(len(corrected)):
            if i % 2 == 0:  # Should be digit
                if corrected[i] in operators:
                    # Replace with most likely digit based on context
                    corrected[i] = '1'  # Conservative choice
            else:  # Should be operator
                if corrected[i] not in operators:
                    # Replace with most likely operator
                    corrected[i] = '+'  # Conservative choice
        
        return ''.join(corrected)
    
    def _is_valid_expression(self, expr):
        if not expr:
            return False
        
        operators = ['+', '-', '×', '÷']
        
        # Should start and end with digit
        if expr[0] in operators or expr[-1] in operators:
            return False
        
        # Check alternating pattern
        for i, char in enumerate(expr):
            if i % 2 == 0:  # Even positions should be digits
                if char in operators:
                    return False
            else:  # Odd positions should be operators
                if char not in operators:
                    return False
        
        return True