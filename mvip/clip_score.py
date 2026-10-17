import torch


def load_clip_model(device: str):
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    return model, processor


@torch.no_grad()
def clip_similarity(ref_frames, gen_frames, model, processor, device: str) -> float:
    def embed(frames):
        inputs = processor(images=frames, return_tensors="pt", padding=True).to(device)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats  # [N, D]

    ref_feats = embed(ref_frames)
    gen_feats = embed(gen_frames)

    sims = (ref_feats * gen_feats).sum(dim=-1)  # [N]
    return sims.mean().item()
