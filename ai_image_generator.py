import os
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline
import gradio as gr

# =============================================
# 🚀 Load Stable Diffusion model
# =============================================
print("🚀 Loading Stable Diffusion model... please wait...")

MODEL_ID = "runwayml/stable-diffusion-v1-5"


try:
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"🔄 Loading model: {model_id} ...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    print(f"✅ Model loaded successfully on {device}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    pipe = None

# =============================================
# 📁 Setup folders and history
# =============================================
os.makedirs("outputs", exist_ok=True)
history_file = "outputs/history.txt"


# =============================================
# 🧠 Image generation function
# =============================================
def generate_image(prompt, steps, guidance, seed):
    if pipe is None:
        return "❌ Model not loaded. Check your terminal for errors.", None

    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
    if seed:
        try:
            seed = int(seed)
            generator.manual_seed(seed)
        except:
            seed = None
    else:
        seed = torch.randint(0, 1000000, (1,)).item()
        generator.manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/{timestamp}_seed{seed}.png"
    image.save(filename)

    # Log prompt
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Prompt: {prompt} | Steps: {steps} | Guidance: {guidance} | Seed: {seed}\n")

    return f"✅ Image generated and saved as {filename}", image


# =============================================
# 📜 View history function
# =============================================
def view_history():
    if not os.path.exists(history_file):
        return "No history yet.", None

    with open(history_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[-10:]  # show last 10 prompts

    prompts = [line.strip() for line in lines]
    images = []

    for line in lines:
        try:
            timestamp = line.split("]")[0].strip("[").strip()
            img_file = next((f for f in os.listdir("outputs") if f.startswith(timestamp)), None)
            if img_file:
                images.append(os.path.join("outputs", img_file))
        except:
            continue

    return "\n".join(prompts), images


# =============================================
# 🎨 Build Gradio interface
# =============================================
with gr.Blocks(title="AI Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 AI Image Generator  
        Turn your imagination into images using **Stable Diffusion v1.5**  
        Images and prompts are auto-saved in the `outputs/` folder.
        """
    )

    with gr.Tab("🧠 Generate Image"):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="e.g. A futuristic cityscape at sunset...", lines=2)
        with gr.Row():
            steps = gr.Slider(10, 50, value=25, step=1, label="Inference Steps")
            guidance = gr.Slider(5, 15, value=7.5, step=0.5, label="Guidance Scale")
            seed = gr.Textbox(label="Seed (optional)", placeholder="Leave blank for random")

        generate_btn = gr.Button("🚀 Generate")
        status = gr.Textbox(label="Status")
        output_image = gr.Image(label="Generated Image")

        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, steps, guidance, seed],
            outputs=[status, output_image],
        )

    with gr.Tab("📜 History"):
        history_text = gr.Textbox(label="Recent Prompts", lines=10)
        history_gallery = gr.Gallery(label="Previous Images", show_label=False, columns=3, height="auto")
        refresh_btn = gr.Button("🔄 Refresh History")

        refresh_btn.click(fn=view_history, outputs=[history_text, history_gallery])

    gr.Markdown("---")
    gr.Markdown("Made with ❤️ by YOU — using Stable Diffusion, PyTorch, and Gradio!")

# =============================================
# 🌐 Launch app
# =============================================
demo.launch(share=True, inbrowser=True)
