import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_bboxes(img, data):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Try to load a default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        
    # Draw text elements
    for elem in data.get("text_elements", []):
        bbox = elem.get("bbox_normalized")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [v for v in bbox]
        abs_bbox = [x1 * width, y1 * height, x2 * width, y2 * height]
        
        type_str = elem.get("type", "other")
        type_colors = {
            "headline": "blue",
            "subheadline": "#42A5F5",
            "cta_button": "green",
            "brand_name": "red",
            "discount_badge": "purple",
            "price_tag": "#E91E63",
            "social_proof": "cyan",
            "legal_text": "#9E9E9E",
            "other": "gray",
        }
        color = type_colors.get(type_str, "gray")
            
        draw.rectangle(abs_bbox, outline=color, width=3)
        draw.text((abs_bbox[0], max(0, abs_bbox[1] - 15)), f"{type_str}: {elem.get('text', '')[:15]}", fill=color, font=font)
        
    # Draw detected elements
    detected = data.get("detected_elements", {})
    colors = {
        "brand_logo": "red",
        "primary_cta": "green",
        "promo_badge": "purple",
        "product_area": "orange",
        "social_proof": "cyan"
    }
    
    for key, color in colors.items():
        val = detected.get(key)
        if val and isinstance(val, dict):
            bbox = val.get("bbox_normalized")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [v for v in bbox]
                abs_bbox = [x1 * width, y1 * height, x2 * width, y2 * height]
                draw.rectangle(abs_bbox, outline=color, width=4)
                conf = val.get('confidence', 0)
                draw.text((abs_bbox[0], max(0, abs_bbox[1] - 15)), f"{key} ({conf:.2f})", fill=color, font=font)
                
    # Draw OpenCV visual elements
    visual_elements = data.get("visual_elements", {}).get("elements", [])
    shape_colors = {
        "card_square": "#FF4444",
        "card_landscape": "#FF8800",
        "card_portrait": "#FFAA00",
        "rounded_card": "#FF6644",
        "circle": "#4488FF",
        "shape": "#88FF44",
    }
    
    for elem in visual_elements:
        bbox = elem.get("bbox_normalized")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [v for v in bbox]
        abs_bbox = [x1 * width, y1 * height, x2 * width, y2 * height]
        
        shape_type = elem.get("shape_type", "shape")
        color = shape_colors.get(shape_type, "#FFFFFF")
        
        draw.rectangle(abs_bbox, outline=color, width=2)
        draw.text((abs_bbox[0] + 2, abs_bbox[1] + 2), f"{shape_type}", fill=color, font=font)

    return img

def create_info_panel(data, img_height):
    panel_width = 400
    panel = Image.new('RGB', (panel_width, img_height), color=(240, 240, 240))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        
    y_text = 10
    
    # Layout Classification
    layout = data.get('layout_classification', data.get('layout_pattern', {}))
    template = layout.get('template', layout.get('type', 'N/A'))
    draw.text((10, y_text), f"Layout: {template}", fill="black", font=title_font)
    y_text += 25
    for k, v in layout.items():
        if k not in ['template', 'type']:
            val_str = f"{v:.3f}" if isinstance(v, float) else str(v)
            draw.text((20, y_text), f"{k}: {val_str}", fill="black", font=font)
            y_text += 18
            
    y_text += 10
    
    # Color Features
    colors_info = data.get('color_features', {})
    if colors_info:
        draw.text((10, y_text), "Color Features:", fill="black", font=title_font)
        y_text += 20
        draw.text((20, y_text), f"Dom: {colors_info.get('dominant_color_name')} ({colors_info.get('dominant_color_hex')})", fill="black", font=font)
        y_text += 18
        draw.text((20, y_text), f"BG: {colors_info.get('background_color_name')} ({colors_info.get('background_color_hex')}) Solid: {colors_info.get('background_is_solid')}", fill="black", font=font)
        y_text += 18
        draw.text((20, y_text), f"Sat: {colors_info.get('saturation_mean')} | Bright: {colors_info.get('brightness_mean')} | Warm: {colors_info.get('color_warmth')}", fill="black", font=font)
        y_text += 25

    # Visual Elements Summary
    vis_summary = data.get('visual_elements', {})
    if vis_summary:
        draw.text((10, y_text), "Visual Elements:", fill="black", font=title_font)
        y_text += 20
        draw.text((20, y_text), f"Total: {vis_summary.get('count', 0)} (Cards: {vis_summary.get('card_count', 0)}, Circles: {vis_summary.get('circle_count', 0)})", fill="black", font=font)
        y_text += 18
        draw.text((20, y_text), f"Grid Like: {vis_summary.get('grid_like')} | Center Y: {vis_summary.get('center_y')}", fill="black", font=font)
        y_text += 25

    # Spatial Features
    draw.text((10, y_text), "Spatial Features:", fill="black", font=title_font)
    y_text += 20
    spatial = data.get('spatial_features', {})
    for k, v in list(spatial.items())[:12]: # limit to fit
        val_str = f"{v:.3f}" if isinstance(v, float) else str(v)
        draw.text((20, y_text), f"{k}: {val_str}", fill="black", font=font)
        y_text += 18
        
    y_text += 10
    
    # Design Conventions
    draw.text((10, y_text), "Design Conventions:", fill="black", font=title_font)
    y_text += 20
    design = data.get('design_conventions', {})
    for k, v in design.items():
        val_str = f"{v:.3f}" if isinstance(v, float) else str(v)
        draw.text((20, y_text), f"{k}: {val_str}", fill="black", font=font)
        y_text += 18

    return panel

def main():
    base_dir = Path(__file__).resolve().parent.parent
    assets_dir = base_dir / "Smadex_Creative_Intelligence_Dataset_FULL" / "assets"
    jsonl_path = base_dir / "cv" / "output" / "creative_prompt_analysis.jsonl"
    output_dir = base_dir / "cv" / "output" / "visualizaciones"
    
    if not jsonl_path.exists():
        print(f"Error: Could not find {jsonl_path}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            creative_id = data.get("creative_id")
            
            # Find image
            img_path = assets_dir / f"{creative_id}.png"
            if not img_path.exists():
                img_path = assets_dir / f"{creative_id}.jpg"
                
            if not img_path.exists():
                print(f"Warning: Image for {creative_id} not found.")
                continue
                
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
                
            # Draw bboxes on image
            annotated_img = draw_bboxes(img.copy(), data)
            
            # Create info panel
            panel = create_info_panel(data, max(img.height, 600))
            
            # Combine image and panel side by side
            new_height = max(img.height, panel.height)
            combined = Image.new('RGB', (img.width + panel.width, new_height), (255, 255, 255))
            combined.paste(annotated_img, (0, 0))
            combined.paste(panel, (img.width, 0))
            
            out_file = output_dir / f"{creative_id}_visualized.png"
            combined.save(out_file)
            count += 1
            
            # if count >= 20: # Visualize first 20 as an example
            #     break
                
    print(f"Successfully generated {count} visualizations in {output_dir}")

if __name__ == "__main__":
    main()
