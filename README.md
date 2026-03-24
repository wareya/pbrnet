# pbrnet

(AI slop -- made by claude webchat -- probably public domain under US law -- model trained only on public domain material)

Oldschool non-latent non-convolutional neural network for making PBR roughness/metalness/ambient occlusion data from color/diffuse/albedo textures. Trained only on public domain material.

**This is *not* a generative AI tool.** It's a granular categorization tool with a model so small it can't *possibly* be memorizing and storing the training data.

Can output ORM, MRO, or separate textures per feature. Feature-generic: can be trained to recognize other analog features as instead of just ORM.

Roughness and AO outputs are decent. Metalness output is somewhat poor, but there's an parameter for hinting at it whether the image has significant metal or not or whether you don't know.

It's python but it uses uv so it's not as awful to use as most python programs.

If you REALLY want, there's a fully in-browser no-server version you can test with, but note that it's like 10x slower. https://wareya.github.io/pbrnet/

```
# convert every file textures/*.dds
$ time uv run python infer.py --dir textures --pattern ".dds" --mro

# or open a web UI for one-at-a-time interactive conversion
$ uv run python infer_gui.py --model pbr_net.pt --port 7860

# if you want to do training:
$ time uv run python build_dataset.py --dir inputs/ --out dataset.npz --ao-augment --rot-flip --tint --multires --blur
$ time uv run python train.py --data dataset.npz --epochs=500
```

<img width="1220" height="598" alt="image" src="https://github.com/user-attachments/assets/9c9f1b43-9116-42f1-90d2-3a709da933a1" />

<img width="1201" height="591" alt="explorer_2026-03-23_16-33-41" src="https://github.com/user-attachments/assets/fe873be6-c26b-4d63-98d7-75f9009601b8" />

<img width="1211" height="596" alt="explorer_2026-03-23_16-13-07" src="https://github.com/user-attachments/assets/ede45f78-3a18-40a7-83f0-226ccde32781" />

<img width="1204" height="596" alt="explorer_2026-03-23_16-13-02" src="https://github.com/user-attachments/assets/6997a727-85aa-4b6f-b121-c9bf1cb29cb2" />

Training data is exclusively CC0 or otherwise public domain material or screenshots of such (you can train your own with your own data, and it trains faster than a convolutional NN would, e.g. in 5~10 minutes):

<img width="1721" height="945" alt="image" src="https://github.com/user-attachments/assets/89c8f9ff-8c32-4a3d-a162-2e537b4f5b25" />
