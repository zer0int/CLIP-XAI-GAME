import sys
import os
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QComboBox, QLabel
from PySide6.QtWidgets import QGridLayout, QLabel
from PySide6.QtGui import QPixmap, QImage, QColor, QIcon
from PySide6.QtCore import Qt, Slot, QTimer, Signal, QSize
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
import random
import warnings
warnings.filterwarnings("ignore")


class ClipApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CLIP Cosine Similarity Memory | Matching Pairs")
        self.initUI()
        self.tiles = {}
        self.selected_pairs = []
        self.firstClick = None
        self.secondClick = None
        self.firstScore = None  # Add this line to store the first score
        self.secondScore = None  # Add this line to store the second score
        self.setMinimumSize(1000, 1200)
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Model selection dropdown
        modelLayout = QHBoxLayout()

        # Model selection dropdown
        self.modelSelect = QComboBox()
        self.modelSelect.addItems(['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'])
        self.modelSelect.setCurrentIndex(2)

        # Add the dropdown to the horizontal layout
        modelLayout.addWidget(self.modelSelect)

        # Label for CLIP Model
        modelLabel = QLabel("CLIP Model 🤖")
        modelLabel.setFixedSize(200,20)  # Set the width of the label to 100 pixels

        # Add the label to the horizontal layout
        modelLayout.addWidget(modelLabel)

        # Add the horizontal layout to the main layout
        layout.addLayout(modelLayout)
        
        # Button for file operations
        self.openImageFolderBtn = QPushButton('Open image folder...')
        self.openImageFolderBtn.setStyleSheet("QPushButton { background-color: yellow; color: black; }")
        layout.addWidget(self.openImageFolderBtn)
        
        # Create a horizontal layout for the difficulty dropdown and the info label
        difficultyLayout = QHBoxLayout()

        # Difficulty selection dropdown
        self.difficultySelect = QComboBox()
        self.difficultySelect.addItems(['Easy (0-1.0)', 'Normal (0-0.8)', 'Hard (0-0.5)'])
        self.difficultySelect.setCurrentIndex(1)  # Set "Normal" as the default

        # Add the dropdown to the horizontal layout
        difficultyLayout.addWidget(self.difficultySelect)

        # Info label with tooltip
        infoLabel = QLabel("Cosine Similarity / Difficulty | 🤔💭 (hover for info)")
        infoLabel.setToolTip("Cosine Similarity = 1 means that two images are, in fact, the same image.\nA value of 0.9 may be expected for two different photos that both depict a cat.\nA cut-off at 0.5, and you'll have to guess what kind of mud-puddle CLIP thinks is median-alike to a pizza. Good luck!")
        infoLabel.setFixedSize(400, 20)  # Set the width of the info label to 100 pixels

        # Add the info label to the horizontal layout
        difficultyLayout.addWidget(infoLabel)

        # Add the horizontal layout to the main layout
        layout.addLayout(difficultyLayout)    
        
      
        
        self.resetBtn = QPushButton('⚠️ Reset App ⚠️')
        self.resetBtn.setStyleSheet("QPushButton { background-color: darkred; color: white; }")
        self.resetBtn.setFixedSize(200, 30)
        layout.addWidget(self.resetBtn)
        
        emptyLabel = QLabel("")
        emptyLabel.setFixedSize(200,10)
        
        # Button to compute cosine similarity
        self.computeBtn = QPushButton('Load images first!')
        self.computeBtn.setEnabled(False)  # Disable the button initially
        #self.computeBtn.setStyleSheet("QPushButton { background-color: lightblue; color: black; }")
        layout.addWidget(self.computeBtn)
        
        # Status Indicator initialization
        self.statusIndicator = QLabel("OK")
        self.statusIndicator.setFixedSize(100, 20)
        self.statusIndicator.setAlignment(Qt.AlignCenter)
        self.updateStatusIndicator("OK")  # Set initial status
        self.statusIndicator.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.statusIndicator)
        
        # Initialize resultsDisplay as a container for the similarity matrix
        self.resultsDisplay = QWidget()
        self.resultsDisplayLayout = QVBoxLayout()
        self.resultsDisplay.setLayout(self.resultsDisplayLayout)
        
        self.resultsDisplay = QWidget()
        self.resultsLayout = QGridLayout()  # Assign a layout to resultsDisplay             
    
        # Add the results display to the main layout
        layout.addWidget(self.resultsDisplay)
    
        self.setLayout(layout)
        
        # Connect buttons
        self.openImageFolderBtn.clicked.connect(self.openImageFolder)
        self.computeBtn.clicked.connect(self.computeCosineSimilarity)
        self.resetBtn.clicked.connect(self.resetApp)
        
    def updateStatusIndicator(self, status):
        """Update the status indicator's text and background color based on the status."""
        color_map = {
            "OK": ("#909090", "black"),  # Light green background, black text
            "READY": ("#90EE90", "black"),  # Light green background, black text
            "RUNNING": ("#FFD700", "black"),  # Yellow background, black text
            "FAIL": ("#FF6347", "white"),  # Tomato background, white text
        }
        color, text_color = color_map.get(status, ("grey", "black"))
        self.statusIndicator.setText(status)
        self.statusIndicator.setStyleSheet(f"background-color: {color}; color: {text_color};")
        
    def resetApp(self):
        # Reset internal state variables
        self.selected_pairs = []
        self.firstClick = None
        self.secondClick = None
        self.firstScore = None
        self.secondScore = None
        self.imageFiles = []
        self.thumbnails = []
    
        # Clear any UI elements that display results or statuses
        for i in reversed(range(self.resultsLayout.count())): 
            self.resultsLayout.itemAt(i).widget().setParent(None)
    
        # Reset input fields and other widgets as necessary
        self.difficultySelect.setCurrentIndex(1)
        self.modelSelect.setCurrentIndex(2)  # Reset to the first model, adjust as needed
        self.updateStatusIndicator("OK")
        self.computeBtn.setEnabled(False)  # Disable the button on reset
        self.computeBtn.setText('Load images first!')
    
        # Add any additional resets here as needed for your application

        
    
    def getFilteredPairs(self, all_pairs):
        try:
            min_val = float(self.minCosineInput.text())
            max_val = float(self.maxCosineInput.text())
        except ValueError:
            print("Invalid range values. Using default range 0 to 1.")
            min_val, max_val = 0.0, 1.0  # Default range
    
        # Filter pairs based on the defined range
        filtered_pairs = [pair for pair in all_pairs if min_val <= pair[2] <= max_val]
        return filtered_pairs        
    
    def openImageFolder(self):
        folderPath = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folderPath:
            # List all files in the selected folder
            files = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
            # Filter out common image formats
            imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            self.imageFiles = [os.path.join(folderPath, f) for f in files if os.path.splitext(f)[1].lower() in imageExtensions]
            if self.imageFiles:  # Check if any images were loaded
                self.computeBtn.setEnabled(True)  # Re-enable the button
                self.computeBtn.setStyleSheet("QPushButton { background-color: lightblue; color: black; }")
                self.computeBtn.setText('Get Cosine Similarity')  # Reset the button text
                print(f"Loaded {len(self.imageFiles)} images.")
                self.updateStatusIndicator("READY")  # Set status back to 'RDY' on success
                QApplication.processEvents()  
            else:
                self.computeBtn.setEnabled(False)  # Keep/Make the button disabled if no images were found
                self.computeBtn.setText('Load images first!')  # Prompt to load images
            
    def createThumbnails(self):
        self.thumbnails = []
        for imagePath in self.imageFiles:
            img = Image.open(imagePath)
            img.thumbnail((100, 100), Image.Resampling.NEAREST)
            qim = ImageQt(img)  # Convert PIL.Image to QImage
            self.thumbnails.append(QPixmap.fromImage(qim))
            
      
    def generateAllValidPairs(self, filtered_pairs):
        # Sort pairs by descending similarity for initial prioritization
        filtered_pairs.sort(key=lambda x: x[2], reverse=True)

        # Initialize containers for valid pairs and used images
        all_valid_pairs = []
        used_images = set()

        # Iterate over each pair, ensuring no image is reused
        for pair in filtered_pairs:
            img1_idx, img2_idx, sim = pair
            if img1_idx in used_images or img2_idx in used_images:
                continue
            all_valid_pairs.append(pair)
            used_images.add(img1_idx)
            used_images.add(img2_idx)

        return all_valid_pairs
        
    def filterPairsInRange(self, all_valid_pairs, min_val, max_val):
        return [pair for pair in all_valid_pairs if min_val <= pair[2] <= max_val]

    def selectRandomPairs(self, filtered_pairs, all_pairs, num_pairs=32):
        # First, try to select pairs within the user's specified range
        selected_pairs = random.sample(filtered_pairs, min(num_pairs, len(filtered_pairs))) if filtered_pairs else []

        # Check if we have enough selected pairs, if not, fill in with additional pairs
        if len(selected_pairs) < num_pairs:
            print("Not enough pairs meet the specified threshold. Filling in with additional pairs.")
            # Exclude already selected pairs
            remaining_pairs = [pair for pair in all_pairs if pair not in selected_pairs]
            # Select additional pairs to meet the requirement
            additional_pairs = random.sample(remaining_pairs, num_pairs - len(selected_pairs))
            selected_pairs.extend(additional_pairs)

        return selected_pairs
        
    
    def createTile(self, img_idx, row, col, sim_score):
        button = QPushButton()
        button.setIcon(QIcon(self.thumbnails[img_idx]))
        button.setIconSize(QSize(100, 100))
        button.setFixedSize(QSize(105, 105))

        pos = (row, col)  # Tuple representing the position in the grid
        self.resultsLayout.addWidget(button, row, col)
        self.tiles[pos] = (img_idx, button)  # Mapping position to image index and button

        button.clicked.connect(lambda checked=False, p=pos: self.onTileClick(p))
     
    def setupTiles(self, selected_pairs):
        self.firstClick = None  # Resetting click tracking
        self.secondClick = None

        # Calculate the total number of positions needed (2 tiles per pair)
        total_positions = len(selected_pairs) * 2

        # Generate a list of all possible positions on the grid
        positions = [(row, col) for row in range(total_positions // 8) for col in range(8)]
    
        # Shuffle the positions to randomize the grid placement
        random.shuffle(positions)

        # Iterating over each selected pair and their new, randomized position
        for pair_index, (img1_idx, img2_idx, sim_score) in enumerate(selected_pairs):
            # Get a shuffled position for each image in the pair
            pos1 = positions[pair_index * 2]
            pos2 = positions[pair_index * 2 + 1]

            # Create and place tiles for each image in the pair using the shuffled positions
            self.createTile(img1_idx, *pos1, sim_score)
            self.createTile(img2_idx, *pos2, sim_score)


    def onTileClick(self, pos):
        # Retrieve image index and button based on position
        if pos in self.tiles:
            img_idx, button = self.tiles[pos]
            button.setIcon(QIcon())

            # Find and display the pair's similarity score, and temporarily remove the icon
            img_idx, button = self.tiles[pos]
            button.setIcon(QIcon())  # Temporarily remove the icon
            for img1_idx, img2_idx, sim_score in self.selected_pairs:
                if img_idx in [img1_idx, img2_idx]:
                    button.setIcon(QIcon())  # Temporarily remove the icon
                    button.setText(f"{sim_score:.2f}")
                    if self.firstClick is None:
                        self.firstClick = pos
                        self.firstScore = sim_score  # Store the first score
                    elif self.secondClick is None and pos != self.firstClick:
                        self.secondClick = pos
                        self.secondScore = sim_score  # Store the second score
                        QTimer.singleShot(2000, self.resetTiles)
                    else:
                        return  # Ignore if two tiles are already selected
                    break
    

    def resetTiles(self):
        if self.firstScore == self.secondScore:  # Check if scores are equal
            for pos in [self.firstClick, self.secondClick]:
                if pos is None:
                    continue
                img_idx, button = self.tiles[pos]
                button.setText("✅")  # Set text to "WIN"
                button.setStyleSheet("background-color: green;")  # Set background to green
                button.setEnabled(False)  # Disable the button
            # Reset clicks and scores but don't reset icons to ensure "WIN" stays
            self.firstClick = None
            self.secondClick = None
            self.firstScore = None
            self.secondScore = None
        else:
            # Original reset logic here (restore icons, clear text, reset clicks and scores)
            for pos in [self.firstClick, self.secondClick]:
                if pos is None:
                    continue
                img_idx, button = self.tiles[pos]
                button.setIcon(QIcon(self.thumbnails[img_idx]))  # Restore the icon
                button.setText("")  # Clear the text
                button.setStyleSheet("")  # Reset any style changes
                button.setEnabled(True)  # Ensure button is enabled again
            self.firstClick = None
            self.secondClick = None
            self.firstScore = None
            self.secondScore = None

        
    
    def computeCosineSimilarity(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Get and cap the min and max values from user input
        # Load the selected model
        self.updateStatusIndicator("RUNNING")  # Update status to 'RUN'
        QApplication.processEvents()
        modelName = self.modelSelect.currentText()
        model, preprocess = clip.load(modelName, device=device)

        self.createThumbnails()

        # Preprocess and compute embeddings for images
        image_embeddings = []
        for imagePath in self.imageFiles:
            image = preprocess(Image.open(imagePath)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = model.encode_image(image)
            image_embeddings.append(image_embedding.squeeze().cpu().numpy())

        # Normalize embeddings to have unit length
        image_embeddings = np.array(image_embeddings)
        norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        normalized_embeddings = image_embeddings / norms

        # Compute cosine similarity - dot product of normalized embeddings
        similarities = np.dot(normalized_embeddings, normalized_embeddings.T)            
                
        # Generate all possible pairs and their similarity values
        all_pairs = [(i, j, similarities[i, j]) for i in range(len(similarities)) for j in range(i + 1, len(similarities))]
        all_valid_pairs = self.generateAllValidPairs(all_pairs)

        # Apply user-defined range
        difficulty_mapping = {
            'Easy (0-1.0)': (0, 1),
            'Normal (0-0.8)': (0, 0.8),
            'Hard (0-0.5)': (0, 0.5),
        }
        difficulty = self.difficultySelect.currentText()
        min_val, max_val = difficulty_mapping[difficulty]
        pairs_in_range = self.filterPairsInRange(all_valid_pairs, min_val, max_val)

        # Randomly select 8 pairs from those in range
        selected_pairs = self.selectRandomPairs(pairs_in_range, all_pairs)
                
        # Now pass these selected pairs to display in the GUI
        self.selected_pairs = selected_pairs
        self.displaySimilarityMatrix(selected_pairs)  
        self.updateStatusIndicator("READY")  # Set status back to 'RDY' on success
        QApplication.processEvents()        
       

    def displaySimilarityMatrix(self, selected_pairs):
        #random.shuffle(selected_pairs)
        self.setupTiles(selected_pairs)
        self.resultsDisplay.setLayout(self.resultsLayout)


    # Placeholder for the click event handler
    def onImageClick(self, img1_idx, img2_idx, sim_score):
        # Logic to handle clicks, reveal similarity scores, check guesses, etc.
        print(f"Clicked on pair with similarity: {sim_score}")

    def onMouseEnter(self, event, label, pixmap):
        label.clear()  # Clear any pixmap that might have been set
        label.setText(text)
        # Retrieve the stored color property again
        color = label.property("color")
        label.setStyleSheet(f"background-color: {color};")

    def onMouseLeave(self, event, label, text):
        label.setPixmap(pixmap)
        # Retrieve the stored color property
        color = label.property("color")
        label.setStyleSheet(f"background-color: {color};")
  
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-size: 12pt; }")
    ex = ClipApp()
    ex.show()
    sys.exit(app.exec())