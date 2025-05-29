from cog import BasePredictor, Input, Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

class Predictor(BasePredictor):
    def setup(self):
        # 加载你的模型
        self.model = ...  # 替换为实际模型加载代码

    def predict(self, image: Path = Input(description="输入图像"), prompt: str = Input(description="发型描述")) -> Path:
        # 打开并转换图像为 RGB
        img = Image.open(image).convert("RGB")
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)
        # 模型预测
        with torch.no_grad():
            output = self.model(img_tensor, prompt)
        # 保存输出图像
        output_image = transforms.ToPILImage()(output.squeeze(0))
        output_path = "/tmp/output.png"
        output_image.save(output_path)
        return Path(output_path)
