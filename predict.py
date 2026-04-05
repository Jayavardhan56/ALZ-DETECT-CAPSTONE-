from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import os
import matplotlib.cm as cm

# ==============================
# MODEL PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "alzheimers_final.keras")

model = None
grad_model = None
last_conv_layer_name_cache = None


# ==============================
# LOAD MODEL
# ==============================

def load_alzheimer_model():
    global model, grad_model, last_conv_layer_name_cache

    if model is None:
        try:
            print("Loading Alzheimer model...")
            print("Model path:", MODEL_PATH)

            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Model file not found.")

            model = load_model(MODEL_PATH, compile=False)

            print("Model loaded successfully")
            print("Input shape:", model.input_shape)

            # Precompute Grad-CAM model
            last_conv_layer_name_cache = None
            for layer in reversed(model.layers):
                try:
                    if len(layer.output.shape) == 4:
                        last_conv_layer_name_cache = layer.name
                        break
                except Exception:
                    continue
                    
            if last_conv_layer_name_cache:
                grad_model = tf.keras.models.Model(
                    [model.inputs], [model.get_layer(last_conv_layer_name_cache).output, model.output]
                )
                print(f"Grad-CAM model initialized with layer: {last_conv_layer_name_cache}")

        except Exception as e:
            print("Model loading failed:", e)
            model = None
            grad_model = None
            last_conv_layer_name_cache = None

    return model


# ==============================
# MRI VALIDATION
# ==============================

def is_brain_mri_like(pil_img):

    # convert to grayscale
    img = pil_img.convert("L").resize((256,256))
    arr = np.array(img).astype("float32") / 255.0

    # 1️⃣ check intensity range (reject blank images)
    if arr.max() - arr.min() < 0.2:
        return False

    h,w = arr.shape

    # 2️⃣ brain usually centered
    center = arr[h//4:3*h//4, w//4:3*w//4]
    border = np.concatenate([
        arr[:h//8,:].flatten(),
        arr[-h//8:,:].flatten(),
        arr[:,:w//8].flatten(),
        arr[:,-w//8:].flatten()
    ])

    if center.mean() < border.mean() + 0.05:
        return False

    # 3️⃣ symmetry check (brain left/right similar)
    left = arr[:, :w//2]
    right = np.fliplr(arr[:, w//2:])

    symmetry = np.mean(np.abs(left - right))

    if symmetry > 0.15:
        return False

    # 4️⃣ texture / edge variation
    edge_var = np.mean(np.abs(np.diff(arr,axis=0))) + np.mean(np.abs(np.diff(arr,axis=1)))

    if edge_var < 0.035:
        return False

    return True


# ==============================
# PREPROCESS IMAGE
# ==============================

def preprocess_image(image_path):

    try:

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        w,h = img.size
        side = min(w,h)

        left = (w-side)//2
        top = (h-side)//2

        img = img.crop((left,top,left+side,top+side))

        if not is_brain_mri_like(img):
            return None,"Uploaded image is not a brain MRI."

        img = img.convert("RGB")
        img = img.resize((224,224))

        arr = np.array(img).astype("float32")

        arr = preprocess_input(arr)

        arr = np.expand_dims(arr,axis=0)

        # Return both the processed array and the original PIL image for Grad-CAM
        return (arr, img), None

    except Exception as e:

        print("Preprocessing error:",e)

        return None,"Image preprocessing failed."


# ==============================
# GRAD-CAM FUNCTIONS
# ==============================

def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        if isinstance(preds, list):
            preds = preds[0]
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        raise ValueError("Could not compute Grad-CAM gradients. Check model architecture.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy(), preds.numpy()[0]

def save_and_display_gradcam(pil_img, heatmap, cam_path, alpha=0.4):
    # Resize original image to 224x224 to match heatmap if needed, or handle sizes
    # Here we assume the input pil_img is the one from preprocess_image (already cropped/resized)
    img_array = np.array(pil_img.convert("RGB")).astype("float32")

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

# ==============================
# PREDICT ALZHEIMER
# ==============================

def predict_alzheimer(image_path):

    try:

        model = load_alzheimer_model()

        if model is None:
            return {"success":False,"message":"Model not loaded"}

        result_data, err = preprocess_image(image_path)

        if result_data is None:
            return {"success":False,"message":err}
        
        img_array, pil_img = result_data

        # Single Pass Optimization: 
        # Use grad_model to get both heatmap and prediction in one forward pass
        if grad_model is not None:
            try:
                heatmap, raw_pred = make_gradcam_heatmap(img_array, grad_model)
                
                # Convert logits to probabilities
                probs = tf.nn.softmax(raw_pred).numpy()
                idx = int(np.argmax(probs))
                
                # Save heatmap alongside the image
                img_dir = os.path.dirname(image_path) or BASE_DIR
                cam_path = os.path.join(img_dir, "cam_" + os.path.basename(image_path))
                save_and_display_gradcam(pil_img, heatmap, cam_path)
            except Exception as gradcam_err:
                print("Grad-CAM error (falling back to standard predict):", gradcam_err)
                raw_pred = model.predict(img_array, verbose=0)[0]
                probs = tf.nn.softmax(raw_pred).numpy()
                idx = int(np.argmax(probs))
        else:
            # Fallback if grad_model fails
            raw_pred = model.predict(img_array, verbose=0)[0]
            probs = tf.nn.softmax(raw_pred).numpy()
            idx = int(np.argmax(probs))

        class_names = [
            "Mild Demented",
            "Moderate Demented",
            "Non-Demented",
            "Very Mild Demented"
        ]

        prediction = class_names[idx]
        confidence = float(probs[idx])

        class_probs = {
            class_names[i]: float(round(float(probs[i]), 4))
            for i in range(len(class_names))
        }

        return {
            "success":True,
            "prediction":prediction,
            "confidence":float(round(float(confidence), 2)),
            "classes":class_probs
        }

    except Exception as e:
        import traceback
        print("Prediction error:", e)
        traceback.print_exc()
        return {
            "success":False,
            "message":"Prediction failed"
        }


# ==============================
# AI SUGGESTIONS
# ==============================

def get_ai_suggestions(prediction_class):

    suggestions = {

        "Non-Demented":
        "No signs of dementia detected. Maintain brain health through exercise, diet, and cognitive stimulation.",

        "Very Mild Demented":
        "Early cognitive changes detected. Regular neurological monitoring is recommended.",

        "Mild Demented":
        "Mild cognitive decline detected. Neurological consultation and cognitive therapy recommended.",

        "Moderate Demented":
        "Moderate dementia detected. Immediate specialist consultation and care planning advised."
    }

    return suggestions.get(prediction_class,"Consult healthcare professional.")