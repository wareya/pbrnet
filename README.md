# pbrnet

(AI slop -- made by claude webchat -- probably public domain under US law -- model trained only on public domain material)

AI texture to PBR material converter/synthesizer.

Oldschool non-latent non-convolutional neural network for making PBR roughness/metalness/ambient occlusion data from color/diffuse/albedo textures. Trained only on public domain material.

**This is *not* a generative AI tool.** It's a granular categorization tool with a model so small it can't *possibly* be memorizing and storing the training data.

Can output ORM, MRO, or separate textures per feature. Feature-generic: can be trained to recognize other granular features and not just ORM, and training is fast, even on CPU.

Roughness and AO outputs are decent. Metalness output is somewhat poor, but there's a parameter for hinting at it whether the image has significant metal or not or whether you don't know.

It's python but it uses uv so it's not as awful to use as most python programs.

If you REALLY want, there's a fully in-browser no-server version you can test with, but note that it's meaningfully slower: https://wareya.github.io/pbrnet/

You can preview the generated textures with this tool here: https://wareya.github.io/pbrnet/pbr_previewer.html

For normals, use: https://github.com/HugoTini/DeepBump

```
# convert every file textures/*.dds
$ time uv run python infer.py --dir textures --pattern ".dds" --mro

# or open a web UI for one-at-a-time interactive conversion
$ uv run python infer_gui.py --model pbr_net.safetensors --port 7860

# if you want to do training:
$ time uv run python build_dataset.py --dir inputs/ --out dataset.npz --ao-augment --rot-flip --tint --multires --blur
$ time uv run python train.py --data dataset.npz --epochs=500
```

<img width="1220" height="598" alt="image" src="https://github.com/user-attachments/assets/9c9f1b43-9116-42f1-90d2-3a709da933a1" />

<img width="1646" height="714" alt="explorer_2026-03-24_17-57-54" src="https://github.com/user-attachments/assets/05ee37fe-27ce-47a0-9822-23691dc97553" />

<img width="1201" height="591" alt="explorer_2026-03-23_16-33-41" src="https://github.com/user-attachments/assets/fe873be6-c26b-4d63-98d7-75f9009601b8" />

<img width="1211" height="596" alt="explorer_2026-03-23_16-13-07" src="https://github.com/user-attachments/assets/ede45f78-3a18-40a7-83f0-226ccde32781" />

<img width="1204" height="596" alt="explorer_2026-03-23_16-13-02" src="https://github.com/user-attachments/assets/6997a727-85aa-4b6f-b121-c9bf1cb29cb2" />

Training data is exclusively CC0 or otherwise public domain material or screenshots of such (you can train your own with your own data, and it trains faster than a convolutional NN would, e.g. in 5~10 minutes):

<img width="1721" height="945" alt="image" src="https://github.com/user-attachments/assets/89c8f9ff-8c32-4a3d-a162-2e537b4f5b25" />

---

## Architecture and Training

Training data was collected from various CC0 and Public Domain sources. Roughly 250 sets of base image to output were used. Most of them came from AmbientCG, but some of them came from OpenGameArt and Sketchhub. The 250 input pairs were checked for data that's set up in ways that works well in 3d rendering but poorly in analysis, like roughness values being too flat or AO never hitting full white, and manually tweaked to work better as training data.

The neural network is a multiscale multilayer perceptron with leaky ReLUs for most of it and a sigmoid-plus-post-scale output phase. The "multiscale" part refers to the way that the inputs numbers are pulled from the image:

For a given output triad (roughness, AO, metal), a varying patch of image is probed. A 9x9 chunk at 100% scale, a 5x5 chunk at 50% scale, a 3x3 chunk at 25% scale, and then 2x2 chunks (-0.5....+0.5) at scales down to and including 1/256. The "size" of the chunks at smaller scales is taken in downscaled pixels, so the 3x3 at 25% scale covers 12x12 pixels worth of area of the original image (really, slightly more because of the edge blur). A global metallicity hint is also generated. This process allows the network to see larger sections of image without having to spend hours learning to ignore irrelevant far-away details like it would have to if it were a convolutional neural network, and it also reduces the risk of overfitting on small data sets. It also allows it to create complex filters over the course of fewer network layers or neurons, because part of the work in doing so is pre-baked into the structure of the inputs.

During training, the patch/chunk data can be augmented as follows: randomly rotate/flip it; randomly tint it; randomly blur it; randomly set the metallicity hint to 0.5; randomly downscale the input texture by 50% or 25% before taking patches; etc. Also, the image is randomly sampled a few hundred times rather than samples being taken for every possible output pixel (number of samples is adjustable); this ensures that training doesn't need to process the entirety of images that are mostly redundant with themselves in terms of possible output values. Images that need more samples (e.g. with a few thin strips or stripes of interesting data) can be provided at a lower resolution, or as multiple copies, or cropped, to increase the chance of hitting the relevant data.

Building a sampled dataset over the 250ish image pairs takes a few minutes. Training the network for 500 epochs takes again a few more minutes, on CPU. Inference takes about a second for particularly large images, again on CPU.

My subjective assessment of the output quality: synthesized roughness is decent, ambient occlusion is OK, and metallicity/metalness is a bit sus but more or less usable without uusing the hint, but OK when using the hint.

Getting quality like this out of such a small network is a reminder that normal, pre-generative AI isn't totally dead.
