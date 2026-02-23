import numpy as np
import cv2
import random

class ImageAugmentor:
    def add_salt_pepper_noise(self, image, amount=0.05):
        noisy_image = np.copy(image)
        num_salt = np.ceil(amount * image.size * 0.5)
        salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[salt_coords[0], salt_coords[1], :] = 255

        num_pepper = np.ceil(amount * image.size * 0.5)
        pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
        return noisy_image

    def add_gaussian_noise(self, image, mean=0, std=25):
        noisy_image = np.copy(image)
        h, w, c = noisy_image.shape
        noise = np.random.normal(mean, std, (h, w, c))
        noisy_image = np.clip(noisy_image + noise, 0, 255)
        return noisy_image.astype(np.uint8)

    def add_shot_noise(self, image, scale=0.1):
        noisy_image = np.copy(image)
        noise = np.random.poisson(scale, image.shape[:2])
        noisy_image = np.clip(noisy_image + noise[..., np.newaxis], 0, 255)
        return noisy_image.astype(np.uint8)

    def add_rain(self, image, slant=-1, drop_length=20, drop_width=1, drop_color=(200, 200, 200), rain_type='torrential', blur_level=3):
        imshape = image.shape
        if slant == -1:
            slant = np.random.randint(-10, 10)
        rain_drops, drop_length = self.generate_random_lines(imshape, slant, drop_length, rain_type)
        image_with_rain = self.rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops, blur_level)
        return image_with_rain

    def add_snow(self, image, snow_point=140, brightness_coefficient=2.5):
        image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        image_HLS = np.array(image_HLS, dtype=np.float64)
        image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] *= brightness_coefficient
        image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
        return cv2.cvtColor(image_HLS.astype(np.uint8), cv2.COLOR_HLS2RGB)

    def change_brightness(self, image, brightness_coefficient):
        image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        image_HLS = np.array(image_HLS, dtype=np.float64)
        image_HLS[:, :, 1] *= brightness_coefficient
        image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
        return cv2.cvtColor(image_HLS.astype(np.uint8), cv2.COLOR_HLS2RGB)

    def add_fog(self, image, fog_coeff=0.5):
        imshape = image.shape
        hw = int(imshape[1] // 3 * fog_coeff)
        haze_list = self.generate_random_blur_coordinates(imshape, hw)
        for haze_points in haze_list:
            image = self.add_blur(image, haze_points[0], haze_points[1], hw, fog_coeff)
        return cv2.blur(image, (hw // 10, hw // 10))

    def adjust_contrast(self, image, alpha=1.0, beta=0):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def add_defocus(self, image, kernel_size=7):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def add_glass_blur(self, image, kernel_size):
        return cv2.blur(image, (kernel_size, kernel_size))

    def add_motion_blur(self, image, kernel_size, style="vert"):
        kernel = np.zeros((kernel_size, kernel_size))
        if style == "vert":
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size) / kernel_size
        elif style == "hori":
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
        return cv2.filter2D(image, -1, kernel)

    def generate_random_lines(self, imshape, slant, drop_length, rain_type):
        area = imshape[0] * imshape[1]
        no_of_drops = area // 500 if rain_type == 'torrential' else area // 700
        drops = [(np.random.randint(0, imshape[1] - slant), np.random.randint(0, imshape[0] - drop_length)) for _ in range(no_of_drops)]
        return drops, drop_length

    def rain_process(self, image, slant, drop_length, drop_color, drop_width, rain_drops, blur_level):
        image_t = image.copy()
        for rain_drop in rain_drops:
            cv2.line(image_t, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color, drop_width)
        
        # Apply blur for rainy effect
        image_t = cv2.blur(image_t, (blur_level, blur_level))

        # Convert to HLS color space
        image_HLS = cv2.cvtColor(image_t, cv2.COLOR_RGB2HLS).astype(np.float64)  # Convert to float64 for brightness scaling
        
        # Scale down brightness channel to simulate shadowy rain effect
        image_HLS[:, :, 1] *= 0.7
        
        # Clip values and convert back to uint8
        image_HLS[:, :, 1] = np.clip(image_HLS[:, :, 1], 0, 255)
        image_HLS = image_HLS.astype(np.uint8)
        
        # Convert back to RGB color space
        return cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)


    def generate_random_blur_coordinates(self, imshape, hw):
        return [(np.random.randint(0, imshape[1] - hw), np.random.randint(0, imshape[0] - hw)) for _ in range(50)]

    def add_blur(self, image, x, y, hw, fog_coeff):
        # Create copies for overlay and output
        overlay = image.copy()
        output = image.copy()  # Make a writable copy for the final output

        # Adjust the blur effect with a circular overlay
        alpha = 0.08 * fog_coeff
        rad = hw // 2
        point = (x + hw // 2, y + hw // 2)
        cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)  # Modify output instead of image directly
        
        return output

