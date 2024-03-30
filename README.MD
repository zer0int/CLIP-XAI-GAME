![Banner](https://github.com/zer0int/CLIP-XAI-GAME/blob/CLIP-vision/examples/clip-logo-xai.jpg?raw=true)

## Playing by AI's rules - Matching Pairs / Memory with CLIP
## Uses image-image Cosine Similarity pairs to determine what is or isn't a match!

Cosine Similarity = 1 means that two images are, in fact, the same image.
A value of 0.8 - 0.9 may be expected for two different photos that both depict a cat.
Set the game to a cut-off value of 0.5, and you'll have to guess what kind of mud-puddle CLIP thinks is of median likeness to a pizza. Good luck!

## Prerequisites / Setup:

- OpenAI / CLIP ([https://github.com/openai/CLIP](https://github.com/openai/CLIP))
- Check / install requirements.txt
- Extract COCO.zip (contains images to play) to any folder
- 250 small images require <2 GB VRAM even with large CLIP models
- Game too hard? Try using only 100 images instead of 250.
- Optional: See "YOLO" folder / readme.txt to use your own photos for the game
- python run-CLIP-matching-pairs.py will launch the GUI

Images via COCO - Common Objects in Context ([https://cocodataset.org/](https://cocodataset.org/))

----

![Screenshot](https://github.com/zer0int/CLIP-XAI-GAME/blob/CLIP-vision/examples/screenshot.jpg?raw=true)
