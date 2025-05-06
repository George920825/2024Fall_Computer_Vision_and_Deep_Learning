import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QGridLayout, QWidget, QFileDialog, QDialog, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.models import vgg19_bn
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
import torchvision.utils as vutils
import random
from torchvision.utils import save_image
from torchvision.utils import make_grid

# Set random seed for reproducibility
# manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        # Number of channels in the training images. For color images this is 3
        nc = 1
        # Size of z latent vector (i.e. size of generator input)
        nz = 100
        # Size of feature maps in generator
        ngf = 64

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        # Number of channels in the training images. For color images this is 3
        nc = 1
        # Size of feature maps in discriminator
        ndf = 64

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
class LossDisplayWindow(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generator and Discriminator Loss")
        self.setGeometry(100, 100, 1200, 600)

        # Create QLabel to display the image
        self.label = QLabel(self)
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

class ImageDisplayWindow(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generated Image")
        self.setGeometry(100, 100, 400, 400)

        # Create QLabel to display the image
        self.label = QLabel(self)
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow with Combined Layout")

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Top layout (Grid layout)
        top_layout = QGridLayout()
        main_layout.addLayout(top_layout)

        # Buttons for the top layout
        self.load_image_button = QPushButton("Load Image")
        self.augmented_images_button = QPushButton("1. Show Augmented Images")
        self.model_structure_button = QPushButton("2. Show Model Structure")
        self.accuracy_loss_button = QPushButton("3. Show Accuracy and Loss")
        self.inference_button = QPushButton("4. Inference")

        # Image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(150, 150)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(Qt.AlignCenter)

        # Prediction label
        self.prediction_label = QLabel("Predicted = None")

        # Adding widgets to the top layout
        top_layout.addWidget(self.load_image_button, 0, 0, 1, 1)
        top_layout.addWidget(self.augmented_images_button, 1, 0, 1, 1)
        top_layout.addWidget(self.model_structure_button, 2, 0, 1, 1)
        top_layout.addWidget(self.accuracy_loss_button, 3, 0, 1, 1)
        top_layout.addWidget(self.inference_button, 4, 0, 1, 1)
        top_layout.addWidget(self.image_label, 0, 1, 3, 1)
        top_layout.addWidget(self.prediction_label, 3, 1, 2, 1)

        # Bottom layout (Vertical layout)
        bottom_layout = QVBoxLayout()
        main_layout.addLayout(bottom_layout)

        # Buttons for the bottom layout
        self.training_images_button = QPushButton("1. Show Training Images")
        self.secondary_model_structure_button = QPushButton("2. Show Model Structure")
        self.training_loss_button = QPushButton("3. Show Training Loss")
        self.secondary_inference_button = QPushButton("4. Inference")

        # Adding buttons to the bottom layout
        bottom_layout.addWidget(self.training_images_button)
        bottom_layout.addWidget(self.secondary_model_structure_button)
        bottom_layout.addWidget(self.training_loss_button)
        bottom_layout.addWidget(self.secondary_inference_button)

        # Connections
        self.load_image_button.clicked.connect(self.load_image)
        self.augmented_images_button.clicked.connect(self.show_augmented_images)
        self.model_structure_button.clicked.connect(self.show_model_structure)
        self.accuracy_loss_button.clicked.connect(self.show_accuracy_and_loss)
        self.inference_button.clicked.connect(self.run_inference)

        self.training_images_button.clicked.connect(self.load_mnist_data)
        self.secondary_model_structure_button.clicked.connect(self.show_model)
        self.training_loss_button.clicked.connect(self.show_training_loss)
        self.secondary_inference_button.clicked.connect(self.create_real_fake_comparison)


        
    def load_image(self):
        # Open file dialog to select an image
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            # Display the image in the QLabel
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.current_image_path = file_path  # Store the path for inference

    # def load_image(self):
    #     file_dialog = QFileDialog()
    #     file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
    #     if file_path:
    #         pixmap = QPixmap(file_path)
    #         self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def show_augmented_images(self):
        # Folder containing images
        folder_path = "../Dataset_CvDl_Hw2/Q1_image/Q1_1/"
        if not os.path.exists(folder_path):
            print("Folder not found: ", folder_path)
            return

        # Load images from folder
        images = []
        file_names = []
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                img_path = os.path.join(folder_path, file_name)
                img = Image.open(img_path)
                images.append(img)
                file_names.append(file_name.split('.')[0])
                if len(images) == 9:
                    break

        # Apply data augmentation
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30)
        ])

        augmented_images = [transform(transforms.ToTensor()(img)).permute(1, 2, 0).numpy() for img in images]

        # Show augmented images with labels
        self.show_images_window(augmented_images, file_names)

    def show_images_window(self, images, labels):
        dialog = QDialog(self)
        dialog.setWindowTitle("Augmented Images with Labels")
        layout = QGridLayout()
        dialog.setLayout(layout)

        for i, (img, label) in enumerate(zip(images, labels)):
            # Convert image to a format that can be displayed in PyQt5
            ax = plt.subplot(3, 3, i + 1)
            ax.axis('off')
            ax.set_title(label, fontsize=8)  # Display the file name
            plt.imshow(img)

        plt.show()

    def show_model_structure(self):
        model = vgg19_bn(pretrained=False, num_classes=10)  # Build VGG19 with BN and 10 output classes
        print("VGG19 Model with Batch Normalization:")
        summary(model, input_size=(3, 32, 32))  # Example input size for the model

    def show_accuracy_and_loss(self):
        # Path to the saved figure
        figure_path = "cvdl_hw2_q1_train_val_accuracy.png"  # Update this path to the location of your saved figure

        if not os.path.exists(figure_path):
            print("Figure not found at:", figure_path)
            return

        # Show the figure in a new dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Training/Validation loss and accuracy")

        layout = QVBoxLayout()
        dialog.setLayout(layout)

        label = QLabel(dialog)
        pixmap = QPixmap(figure_path)
        label.setPixmap(pixmap.scaled(1000, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(label)

        dialog.exec_()

    def run_inference(self):
        # Load the trained model
        model_path = "cvdl_hw2_q1.pth"  # Replace with the path to your trained model
        if not os.path.exists(model_path):
            print("Model not found at:", model_path)
            return
        
        model = vgg19_bn(pretrained=False, num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()

        # Check if an image has been loaded
        if not hasattr(self, "current_image_path") or not os.path.exists(self.current_image_path):
            print("No image loaded!")
            return

        # Preprocess the image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        image = Image.open(self.current_image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = nn.functional.softmax(output, dim=1).squeeze().numpy()  # Apply softmax

        # Get predicted class
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]  # Replace with your model's class names if different
        predicted_class = probabilities.argmax()
        self.prediction_label.setText(f"Predicted = {class_names[predicted_class]}")  # Update label on GUI

        # Show probability distribution
        self.show_histogram(probabilities)

    # def show_histogram(self, probabilities):
    #     plt.figure(figsize=(8, 6))
    #     plt.bar(range(len(probabilities)), probabilities, color='blue')
    #     plt.xlabel("Class Index")
    #     plt.ylabel("Probability")
    #     plt.title("Probability Distribution of Predictions")
    #     plt.show()

    def show_histogram(self, probabilities):
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]  # Replace with your model's class names if different

        plt.figure(figsize=(10, 6))
        plt.bar(class_names, probabilities, color='blue')
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title("Probability of Each Class")

        # Annotate the bar with probability values
        for i, prob in enumerate(probabilities):
            plt.text(i, prob + 0.01, f'{prob:.3f}', ha='center', fontsize=10)

        plt.ylim(0, 1.1)  # Adjust y-axis to leave space for annotations
        plt.show()

    def load_mnist_data(self):
        dataroot = "../Dataset_CvDl_Hw2/Q2_images/data"  # Path to the MNIST dataset

        batch_size = 128
        image_size = 64

        transform_original = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5)),
        ])
        transform_augmented = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(),
            transforms.RandomRotation(degrees=60),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5)),
        ])

        # Create the dataset
        dataset_original = dset.ImageFolder(root=dataroot, transform=transform_original)
        dataset_augmented = dset.ImageFolder(root=dataroot, transform=transform_augmented)

        # Create the dataloader
        dataloader_original = DataLoader(dataset_original, batch_size=batch_size, shuffle=True)
        dataloader_augmented = DataLoader(dataset_augmented, batch_size=batch_size, shuffle=True)

        # Plot some training images
        images_original = next(iter(dataloader_original))
        images_augmented = next(iter(dataloader_augmented))

        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Training Dataset (Original)")
        plt.imshow(np.transpose(vutils.make_grid(images_original[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Training Dataset (Augmented)")
        plt.imshow(np.transpose(vutils.make_grid(images_augmented[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()

    def show_model(self):
        # custom weights initialization called on ``netG`` and ``netD``
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        # Number of GPUs available. Use 0 for CPU mode.
        ngpu = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Create the generator
        netG = Generator(ngpu).to(device)

        # Handle multi-GPU if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the ``weights_init`` function to randomly initialize all weights
        #  to ``mean=0``, ``stdev=0.02``.
        netG.apply(weights_init)

        # Print the model
        print(netG)

        # Create the Discriminator
        netD = Discriminator(ngpu).to(device)

        # Handle multi-GPU if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the ``weights_init`` function to randomly initialize all weights
        # like this: ``to mean=0, stdev=0.2``.
        netD.apply(weights_init)

        # Print the model
        print(netD)

    def show_training_loss(self):
        # Path to the saved training loss figure
        loss_figure_path = "cvdl_hw2_q2_loss_plot.png"

        # Create and show the new window
        self.loss_window = LossDisplayWindow(loss_figure_path)
        self.loss_window.show()

    def create_real_fake_comparison(self):
        # Parameters
        batch_size = 64
        image_size = 64
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        generator_model_path = "cvdl_hw2_q2_best_generator.pth"
        # Load the pre-trained generator model
        netG = Generator(ngpu=0).to(device)
        netG.load_state_dict(torch.load(generator_model_path, map_location=device, weights_only=True))
        netG.eval()

        # Load real images from MNIST dataset
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset_path = "../Dataset_CvDl_Hw2/Q2_images/data"
        dataset = dset.ImageFolder(root=dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Get a batch of real images
        real_images, _ = next(iter(dataloader))
        
        # Generate a batch of fake images
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        with torch.no_grad():
            fake_images = netG(noise).cpu()

        # Create grids for real and fake images
        real_grid = make_grid(real_images[:64], nrow=8, normalize=True, padding=2)
        fake_grid = make_grid(fake_images[:64], nrow=8, normalize=True, padding=2)

        # Plot the grids
        plt.figure(figsize=(12, 6))

        # Real Images
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(real_grid.numpy(), (1, 2, 0)))

        # Fake Images
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(fake_grid.numpy(), (1, 2, 0)))

        # Show the plot
        plt.show()

# Main application
app = QApplication(sys.argv)

# Create and show main window
main_window = MainWindow()
main_window.show()

sys.exit(app.exec_())
