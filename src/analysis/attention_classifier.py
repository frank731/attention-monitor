import base64
import numpy as np
import cv2

def load_pose_model(checkpoint_path):
    """Load emotion detection model from checkpoint"""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    
    class FaceResNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            # Load pretrained ResNet
            weights = ResNet18_Weights.DEFAULT
            self.resnet = resnet18(weights=weights)
            
            # Convert grayscale to RGB by repeating channel
            self.grayscale_to_rgb = lambda x: x.repeat(1, 3, 1, 1)
            
            # Modify final layer
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_features, num_classes)

        def forward(self, x):
            x = self.grayscale_to_rgb(x)
            return self.resnet(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceResNet(num_classes=6).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def predict_pose(frame, model, device):
    """
    Predict emotion from a cv2 frame in memory
    Args:
        frame: cv2 BGR image
        model: loaded pytorch model
        device: torch device
    Returns:
        (predicted_class_idx, confidence, emotion_name)
    """
    import cv2
    import torch
    from torchvision import transforms
    from PIL import Image
    
    # Convert BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert to PIL Image
    image = Image.fromarray(gray)
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probabilities[0][pred_class].item()
    
    emotions = []
    return pred_class, confidence, emotions[pred_class]


def encode_numpy_image(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

class AttentionStatus:
    OnPhone = "ONPHONE"
    Distracted = "DISTRACTED"
    Focused = "FOCUSED"

STATUS_TO_TEXT = {
    AttentionStatus.OnPhone: "On Phone",
    AttentionStatus.Distracted: "Distracted",
    AttentionStatus.Focused: "Attentive"
}

class AttentionClassifier:
    def __init__(self):
        self.model = None

    def classify_attention(self, image: np.ndarray) -> AttentionStatus:
        return self.prompt_model_estimator_mock_test(image)


    def prompt_model_estimator_mock_test(self, image: np.ndarray) -> AttentionStatus:
        import openai
        if not self.model:
            self.model = openai.Client()

        base64_image = encode_numpy_image(image)
        response = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """This is a photo of someone during a lecture. Does it look like this person is:
(a) Looking at the camera and paying attention
(b) Looking down or around and not paying attention or distracted
(c) Using their phone

Answer with just the letter."""},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low" }}
                    ]
                }
            ]
        )

        letter = response.choices[0].message.content.lower().split()[0]
        if "c" in letter:
            return AttentionStatus.OnPhone
        if "b" in letter:
            return AttentionStatus.Distracted

        return AttentionStatus.Focused
    

    def prompt_model_estimator_custom_model(self, image):
        if not self.model:
            self.model = load_pose_model("ml/checkpoint.pth")

        return predict_pose(image, self.model, "cuda")
