1. Download YOLO V7 weights and put them in this folder: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.weights

2. Edit the yolo-square-crop.py file by setting your image folder to be processed + the output folder for YOLO crops

			~~ WARNING ~~ 		~~ WARNING ~~ 		~~ WARNING ~~ 		~~ WARNING ~~ 
Will loop through the folder indefinitely, thus allowing YOLO a chance to stochastically 'see' another object in the same image + make a crop of that (saved with different filename), too.
You'll want to CTRL+C terminate that loop at *some* point, though!

3. Edit the cleanup-delete-non-square.py by setting your YOLO crop outputs folder
   -> Deletes non-square crops created by the YOLO script, as CLIP takes square width==height images as input. Non-square will be "squished" into being square, with bad results.

4. Done! You can now use the CLIP cosine similarity memory game with your holiday photos (or whatever)!
   -> You can also use any arbitrary dataset, for example: cocodataset.org or check kaggle.com/datasets?search=image


NOTE: Keep in mind CLIP has to process all those images for computing cosine similarity. Like, maybe don't use THOUSANDS of images. Although ~250 small images requires ~2 GB VRAM even with a large CLIP model.