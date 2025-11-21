import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(image_path, mask_path=None, alpha=0.5, mask_color=(255, 0, 0)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 如果有mask
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # 尺寸不匹配时调整
        if mask.shape != image_rgb.shape[:2]:
            mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

        # 转布尔掩码
        mask_bool = mask > 0

        # 颜色层
        overlay_color = np.full_like(image_rgb, mask_color, dtype=np.uint8)

        # 混合整张图，然后在 mask 区域替换
        blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay_color, alpha, 0)
        image_rgb[mask_bool] = blended[mask_bool]

    # 显示
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()
