import torch
import os

def test_local_hub():
    print("Testing RADSeg Torch Hub integration locally...")
    try:
        # Load from the current directory ('.') as the source
        # This simulates how torch.hub would work when pointing to a repo
        model = torch.hub.load('.', 'radseg_encoder', source='local', 
                               model_version="c-radio_v3-b", 
                               lang_model="siglip2", 
                               device='cpu') # Use cpu for simple check
        
        print("Successfully loaded RADSegEncoder via torch.hub!")
        print(f"Model Class: {type(model)}")
        
        # Simple dry run: check if we can encode a dummy image
        dummy_img = torch.randn(1, 3, 224, 224)
        print("Testing image encoding...")
        with torch.no_grad():
            feat_map = model.encode_image_to_feat_map(dummy_img)
        print(f"Feature map shape: {feat_map.shape}")
        print("Integration test PASSED.")
        
    except Exception as e:
        print(f"Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_hub()
