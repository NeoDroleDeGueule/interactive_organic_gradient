import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import json
import colorsys
import random
import math


def smooth_mask(mask, smoothness):
    # smoothness >= 0.1 : plus grand -> plus doux
    eps = 1e-8
    s = max(0.1, float(smoothness))
    return 1.0 - (1.0 - mask) ** (1.0 / (s + eps))

def hex_to_rgb(hexstr):
    return (int(hexstr[1:3], 16), int(hexstr[3:5], 16), int(hexstr[5:7], 16))

def interpolate_color_2d_rgb(stops, x, y):
    total_weight = 0.0
    r_total = g_total = b_total = 0.0
    for stop in stops:
        sx, sy = stop["x"], stop["y"]
        r, g, b = hex_to_rgb(stop["color"])
        dist_sq = (x - sx) ** 2 + (y - sy) ** 2
        if dist_sq == 0:
            return (r, g, b)
        weight = 1.0 / (dist_sq + 1e-6)
        r_total += r * weight
        g_total += g * weight
        b_total += b * weight
        total_weight += weight
    return (int(r_total / total_weight), int(g_total / total_weight), int(b_total / total_weight))

def interpolate_color_2d_hsv(stops, x, y):
    total_weight = 0.0
    sum_cos = sum_sin = s_total = v_total = 0.0
    for stop in stops:
        sx, sy = stop["x"], stop["y"]
        r, g, b = hex_to_rgb(stop["color"])
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
        dist_sq = (x - sx) ** 2 + (y - sy) ** 2
        if dist_sq == 0:
            return (r, g, b)
        weight = 1.0 / (dist_sq + 1e-6)
        sum_cos += np.cos(2.0 * np.pi * h) * weight
        sum_sin += np.sin(2.0 * np.pi * h) * weight
        s_total += s * weight
        v_total += v * weight
        total_weight += weight
    if total_weight == 0:
        return (0, 0, 0)
    avg_cos = sum_cos / total_weight
    avg_sin = sum_sin / total_weight
    avg_h = (np.arctan2(avg_sin, avg_cos) / (2.0 * np.pi)) % 1.0
    avg_s = s_total / total_weight
    avg_v = v_total / total_weight
    rf, gf, bf = colorsys.hsv_to_rgb(avg_h, avg_s, avg_v)
    return (int(rf * 255), int(gf * 255), int(bf * 255))


class InteractiveOrganicGradientNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "blob_shape": (["circle", "radial", "donut", "rectangle", "horizontal_stripe", "vertical_stripe", "diamond", "triangle", "star", "blob_random", "spore"],),
                "blur_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blob_size": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "blob_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radial_smoothness": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1, "display": "slider"}),
                "gradient_data": (
                    "STRING",
                    {
                        "default": '[{"x":0.2,"y":0.5,"color":"#ff3300"},{"x":0.8,"y":0.5,"color":"#00ffe1"}]',
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "palette_image", "palette_hex")
    FUNCTION = "generate"
    CATEGORY = "Custom Nodes/Interactive"

    def generate(self, width, height, blob_shape, blur_strength, gradient_data, blob_size, blob_opacity, radial_smoothness):
        
        # Lecture du JSON du gradient
        try:
            gradient_points = json.loads(gradient_data)
        except Exception:
            gradient_points = [{"x": 0.1, "y": 0.5, "color": "#ff0000"}, {"x": 0.9, "y": 0.5, "color": "#0000ff"}]

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        def interpolate_color_2d(stops, x, y, smoothness):
            total_weight = 0
            r_total = g_total = b_total = 0
            for stop in stops:
                sx, sy = stop["x"], stop["y"]
                color = stop["color"]
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                dist_sq = (x - sx) ** 2 + (y - sy) ** 2
                weight = 1.0 / ((dist_sq + 0.001) ** smoothness)
                r_total += r * weight
                g_total += g * weight
                b_total += b * weight
                total_weight += weight
            if total_weight == 0:
                return (0, 0, 0)
            return (
                int(r_total / total_weight),
                int(g_total / total_weight),
                int(b_total / total_weight),
            )

        # --- Définition de base du gradient ---
        def linear_gradient(x):
            stops = sorted(gradient_data, key=lambda s: s["x"])
            for i in range(len(stops) - 1):
                if stops[i]["x"] <= x <= stops[i + 1]["x"]:
                    t = (x - stops[i]["x"]) / (stops[i + 1]["x"] - stops[i]["x"])
                    c1 = tuple(int(stops[i]["color"][j:j + 2], 16) for j in (1, 3, 5))
                    c2 = tuple(int(stops[i + 1]["color"][j:j + 2], 16) for j in (1, 3, 5))
                    return interpolate_color(c1, c2, t)
            c = stops[-1]["color"]
            return [int(c[j:j + 2], 16) for j in (1, 3, 5)]

        for stop in gradient_points:
            sx, sy = stop["x"], stop["y"]
            x, y = int(sx * width), int(sy * height)
            rel_x, rel_y = x / width, y / height
            color = interpolate_color_2d(gradient_points, rel_x, rel_y, radial_smoothness)

            temp_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(temp_img, "RGBA")

            w, h = int(blob_size * width), int(blob_size * height)
            shape_color = (*color, 255)

            # ==== SHAPES =====================================================

            if blob_shape == "circle":
                draw.ellipse((x - w, y - h, x + w, y + h), fill=shape_color)

            # -------------------------------------------------------------
            # 🔸 Forme : RADIAL
            # -------------------------------------------------------------
            elif blob_shape == "radial":
                radius = int(blob_size * min(width, height))
                size = radius * 2

                yy, xx = np.mgrid[-radius:radius, -radius:radius]
                dist = np.sqrt(xx**2 + yy**2) / float(radius)
                dist = np.clip(dist, 0.0, 1.0)

                # masque brut
                mask = 1.0 - dist

                # application du lissage correct
                mask = smooth_mask(mask, radial_smoothness)

                alpha = (mask * 255.0).astype(np.uint8)
                grad = Image.fromarray(alpha, mode="L")

                blob = Image.new("RGBA", (size, size), (*color, 255))
                blob.putalpha(grad)

                # Ne pas coller ici, mais créer temp_img pour que blur et opacity s'appliquent
                temp_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                temp_img.paste(blob, (x - radius, y - radius), blob)

            # -------------------------------------------------------------
            # 🔹 Forme : DONUT (trou au centre, sensible à blur_strength et blob_opacity)
            # -------------------------------------------------------------
            elif blob_shape == "donut":
                radius = int(blob_size * min(width, height))
                size = radius * 2

                # Créer une image temporaire pour le donut
                temp_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(temp_img, "RGBA")

                # Dessiner un cercle extérieur opaque
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 255))

                # Dessiner un cercle intérieur transparent (pour faire le trou)
                inner_radius = radius * 0.5  # 50% de la taille du cercle extérieur
                draw.ellipse((x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius), fill=(0, 0, 0, 0))

                # Ne pas coller ici, mais laisser le code général s'occuper de blur et opacity

            # -------------------------------------------------------------
            elif blob_shape == "rectangle":
                draw.rectangle((x - w, y - h, x + w, y + h), fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "horizontal_stripe":
                # Bande horizontale sur toute la largeur du canvas
                band_height = int(blob_size * height)
                y0 = max(0, y - band_height // 2)
                y1 = min(height, y + band_height // 2)
                draw.rectangle((0, y0, width, y1), fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "vertical_stripe":
                # Bande verticale sur toute la hauteur du canvas
                band_width = int(blob_size * width)
                x0 = max(0, x - band_width // 2)
                x1 = min(width, x + band_width // 2)
                draw.rectangle((x0, 0, x1, height), fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "diamond":
                points = [(x, y - h), (x + w, y), (x, y + h), (x - w, y)]
                draw.polygon(points, fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "triangle":
                points = [(x, y - h), (x + w, y + h), (x - w, y + h)]
                draw.polygon(points, fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "star":
                points = []
                spikes = 5
                outer_r = w
                inner_r = w * 0.4
                for i in range(spikes * 2):
                    angle = math.pi / spikes * i
                    r = outer_r if i % 2 == 0 else inner_r
                    px = x + math.cos(angle) * r
                    py = y + math.sin(angle) * r
                    points.append((px, py))
                draw.polygon(points, fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "blob_random":
                points = []
                num_points = random.randint(6, 12)
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    r = w * (0.7 + 0.3 * random.random())
                    px = x + math.cos(angle) * r
                    py = y + math.sin(angle) * r
                    points.append((px, py))
                draw.polygon(points, fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "spore":
                radius = int(blob_size * min(width, height))
                temp_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(temp_img, "RGBA")
                cx, cy = x, y
                points = []
                num_points = 120
                noise_strength = 0.25
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    noise = (random.random() - 0.5) * 2 * noise_strength
                    r = radius * (1 + noise)
                    px = cx + r * math.cos(angle)
                    py = cy + r * math.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=shape_color)

                spore_array = np.array(temp_img)
                alpha_layer = spore_array[..., 3].astype(np.float32) / 255.0
                y_indices, x_indices = np.indices((height, width))
                dist = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
                dist /= radius
                fade = np.clip(1.0 - dist ** 1.5, 0, 1)
                alpha_layer *= fade
                spore_array[..., 3] = (alpha_layer * 255).astype(np.uint8)
                temp_img = Image.fromarray(spore_array, "RGBA")

            # ================================================================

            if blur_strength > 0:
                blur_val = int((width + height) / 2 * blur_strength / 6)
                if blur_val > 0:
                    temp_img = temp_img.filter(ImageFilter.GaussianBlur(radius=blur_val))

            # Appliquer l’opacité
            temp_np = np.array(temp_img)
            temp_np[..., 3] = (temp_np[..., 3].astype(np.float32) * blob_opacity).astype(np.uint8)
            temp_img = Image.fromarray(temp_np, "RGBA")

            # Fusion dans l’image principale
            base_np = np.array(img).astype(np.float32)
            overlay_np = np.array(temp_img).astype(np.float32)
            alpha_overlay = overlay_np[..., 3:] / 255.0
            alpha_base = base_np[..., 3:] / 255.0
            out_rgb = overlay_np[..., :3] * alpha_overlay + base_np[..., :3] * (1 - alpha_overlay)
            out_alpha = alpha_overlay + alpha_base * (1 - alpha_overlay)
            out = np.dstack([out_rgb, out_alpha * 255]).astype(np.uint8)
            img = Image.fromarray(out, mode="RGBA")

        # Tensor final
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]

        # Palette
        colors = [tuple(int(p["color"][i:i + 2], 16) for i in (1, 3, 5)) for p in gradient_points]
        palette_img = Image.new("RGB", (len(colors) * 20, 20))
        draw_pal = ImageDraw.Draw(palette_img)
        for i, color in enumerate(colors):
            draw_pal.rectangle([i * 20, 0, (i + 1) * 20, 20], fill=color)
        palette_tensor = torch.from_numpy(np.array(palette_img).astype(np.float32) / 255.0)[None,]
        palette_hex = ", ".join([p["color"] for p in gradient_points])

        return (img_tensor, palette_tensor, palette_hex)

class ImageBlendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mode": ([
                    "normal", "multiply", "screen", "overlay",
                    "add", "subtract", "difference", "lighten", "darken"
                ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "Image/Blend"

    def add_alpha(self, arr):
        """Ajoute un canal alpha (opaque) si absent"""
        if arr.shape[2] == 3:
            alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=2)
        return arr

    def resize_and_crop_to_cover(self, arr, target_h, target_w):
        """Resize + crop pour que arr recouvre entièrement (cover) la cible"""
        h, w = arr.shape[:2]

        # Calcul du scale factor (cover)
        scale = max(target_w / w, target_h / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize avec PIL
        img_pil = Image.fromarray((arr * 255).astype(np.uint8)) if arr.dtype != np.uint8 else Image.fromarray(arr)
        img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
        arr_resized = np.array(img_resized).astype(np.float32) / 255.0

        # Crop centré
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        arr_cropped = arr_resized[start_y:start_y + target_h, start_x:start_x + target_w, :]

        return arr_cropped

    def blend_mode(self, a, b, mode):
        if mode == "normal":
            return b
        elif mode == "multiply":
            return a * b
        elif mode == "screen":
            return 1 - (1 - a) * (1 - b)
        elif mode == "overlay":
            return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        elif mode == "add":
            return np.clip(a + b, 0, 1)
        elif mode == "subtract":
            return np.clip(a - b, 0, 1)
        elif mode == "difference":
            return np.abs(a - b)
        elif mode == "lighten":
            return np.maximum(a, b)
        elif mode == "darken":
            return np.minimum(a, b)
        else:
            return b

    def blend(self, image_a, image_b, mode, opacity):
        arr_a = image_a[0].cpu().numpy()
        arr_b = image_b[0].cpu().numpy()

        # Normalisation [0,1]
        if arr_a.max() > 1.0: arr_a = arr_a / 255.0
        if arr_b.max() > 1.0: arr_b = arr_b / 255.0

        # Ajout alpha si manquant
        arr_a = self.add_alpha(arr_a)
        arr_b = self.add_alpha(arr_b)

        # Resize image_b pour couvrir image_a
        arr_b = self.resize_and_crop_to_cover(arr_b, arr_a.shape[0], arr_a.shape[1])

        # Blend par mode
        blended_rgb = self.blend_mode(arr_a[..., :3], arr_b[..., :3], mode)

        # Gestion alpha avec opacity
        alpha_a = arr_a[..., 3:4]
        alpha_b = arr_b[..., 3:4] * opacity
        out_rgb = (1 - alpha_b) * arr_a[..., :3] + alpha_b * blended_rgb

        # Clamp [0,1]
        out_rgb = np.clip(out_rgb, 0, 1)

        # Conversion tensor
        out_tensor = torch.from_numpy(out_rgb).unsqueeze(0).float()

        return (out_tensor,)



NODE_CLASS_MAPPINGS = {
    "InteractiveOrganicGradientNode": InteractiveOrganicGradientNode,
    "ImageBlendNode": ImageBlendNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractiveOrganicGradientNode": "🍄Interactive Organic Gradient",
    "ImageBlendNode": "🍄Image Blend"
}
