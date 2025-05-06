import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDialog,
    QPushButton, QLabel, QSpinBox, QGroupBox, QTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt

class MatrixDialog(QDialog):
    # Dialog to display the matrix
    def __init__(self, matrix, title):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 50, 300, 100)

        # Format the matrix as a string
        matrix_str = np.array2string(matrix)

        # Set up the layout and label to display the matrix
        layout = QVBoxLayout()
        label = QLabel(f"{title}:\n{matrix_str}")
        layout.addWidget(label)
        self.setLayout(layout)

    def keyPressEvent(self, event):
        # Override keyPressEvent to close the dialog on any key press.
        self.accept()  # Close the dialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow - cvdlhw1.ui")
        self.setGeometry(100, 100, 1000, 600)

        # Initialize a list to hold image file paths
        self.image_paths = []

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Load Image Section
        load_image_group = QGroupBox("Load Image")
        load_image_layout = QVBoxLayout()
        
        load_folder_button = QPushButton("Load folder")
        load_folder_button.setFixedSize(100, 40)
        load_folder_button.clicked.connect(self.load_folder)  # Connect button to function
        
        load_image_l_button = QPushButton("Load Image_L")
        load_image_l_button.setFixedSize(100, 40)
        load_image_l_button.clicked.connect(self.load_imageL)
        
        load_image_r_button = QPushButton("Load Image_R")
        load_image_r_button.setFixedSize(100, 40)
        load_image_r_button.clicked.connect(self.load_imageR)
        
        load_image_layout.addWidget(load_folder_button, alignment=Qt.AlignCenter)
        load_image_layout.addWidget(load_image_l_button, alignment=Qt.AlignCenter)
        load_image_layout.addWidget(load_image_r_button, alignment=Qt.AlignCenter)
        load_image_group.setLayout(load_image_layout)

        # Calibration Section
        calibration_group = QGroupBox("1. Calibration")
        calibration_layout = QVBoxLayout()
        
        find_corners_button = QPushButton("1.1 Find corners")
        find_corners_button.setFixedSize(150, 40)
        find_corners_button.clicked.connect(self.find_corners)  # Connect button to function
        
        find_intrinsic_button = QPushButton("1.2 Find intrinsic")
        find_intrinsic_button.setFixedSize(150, 40)
        find_intrinsic_button.clicked.connect(self.find_intrinsic)
        
        
        find_extrinsic_group = QGroupBox("1.3 Find extrinsic")
        find_extrinsic_layout = QHBoxLayout()
        self.spin_box = QSpinBox()
        self.spin_box.setRange(1,15)
        self.spin_box.setSingleStep(1)
        self.spin_box.setFixedWidth(50)
        
        find_extrinsic_button = QPushButton("1.3 Find extrinsic")
        find_extrinsic_button.setFixedSize(120, 40)
        find_extrinsic_button.clicked.connect(self.find_extrinsic)
        
        find_extrinsic_layout.addWidget(self.spin_box)
        # find_extrinsic_layout.addWidget(self.image_number_box)
        find_extrinsic_layout.addWidget(find_extrinsic_button)
        find_extrinsic_group.setLayout(find_extrinsic_layout)
        
        find_distortion_button = QPushButton("1.4 Find distortion")
        find_distortion_button.setFixedSize(150, 40)
        find_distortion_button.clicked.connect(self.find_distortion)
        
        show_result_button = QPushButton("1.5 Show result")
        show_result_button.setFixedSize(150, 40)
        show_result_button.clicked.connect(self.show_result)
        
        calibration_layout.addWidget(find_corners_button, alignment=Qt.AlignCenter)
        calibration_layout.addWidget(find_intrinsic_button, alignment=Qt.AlignCenter)
        calibration_layout.addWidget(find_extrinsic_group, alignment=Qt.AlignCenter)
        calibration_layout.addWidget(find_distortion_button, alignment=Qt.AlignCenter)
        calibration_layout.addWidget(show_result_button, alignment=Qt.AlignCenter)
        calibration_group.setLayout(calibration_layout)

        # Augmented Reality Section
        ar_group = QGroupBox("2. Augmented Reality")
        ar_layout = QVBoxLayout()
        
        self.text_edit = QTextEdit()
        self.text_edit.setFixedSize(200, 150)
        
        show_words_board_button = QPushButton("2.1 show words on board")
        show_words_board_button.setFixedSize(180, 40)
        show_words_board_button.clicked.connect(self.show_words_board)
        
        show_words_vertical_button = QPushButton("2.2 show words vertical")
        show_words_vertical_button.setFixedSize(180, 40)
        show_words_vertical_button.clicked.connect(self.show_words_vertical)
        
        ar_layout.addWidget(self.text_edit, alignment=Qt.AlignCenter)
        ar_layout.addWidget(show_words_board_button, alignment=Qt.AlignCenter)
        ar_layout.addWidget(show_words_vertical_button, alignment=Qt.AlignCenter)
        ar_group.setLayout(ar_layout)

        # Stereo Disparity Map Section
        stereo_group = QGroupBox("3. Stereo disparity map")
        stereo_layout = QVBoxLayout()
        
        stereo_disparity_button = QPushButton("3.1 stereo disparity map")
        stereo_disparity_button.setFixedSize(150, 40)
        stereo_disparity_button.clicked.connect(self.stereo_disparity)
        
        stereo_layout.addWidget(stereo_disparity_button, alignment=Qt.AlignCenter)
        stereo_group.setLayout(stereo_layout)

        # SIFT Section
        sift_group = QGroupBox("4. SIFT")
        sift_layout = QVBoxLayout()
        
        load_image1_button = QPushButton("Load Image1")
        load_image1_button.setFixedSize(150, 40)
        load_image1_button.clicked.connect(self.load_image1)
        
        load_image2_button = QPushButton("Load Image2")
        load_image2_button.setFixedSize(150, 40)
        load_image2_button.clicked.connect(self.load_image2)
        
        sift_keypoints_button = QPushButton("4.1 Keypoints")
        sift_keypoints_button.setFixedSize(150, 40)
        sift_keypoints_button.clicked.connect(self.sift_keypoints)
        
        matched_keypoints_button = QPushButton("4.2 Matched Keypoints")
        matched_keypoints_button.setFixedSize(150, 40)
        matched_keypoints_button.clicked.connect(self.matched_keypoints)
        
        sift_layout.addWidget(load_image1_button, alignment=Qt.AlignCenter)
        sift_layout.addWidget(load_image2_button, alignment=Qt.AlignCenter)
        sift_layout.addWidget(sift_keypoints_button, alignment=Qt.AlignCenter)
        sift_layout.addWidget(matched_keypoints_button, alignment=Qt.AlignCenter)
        sift_group.setLayout(sift_layout)

        # Add all groups to the main layout
        main_layout.addWidget(load_image_group)
        main_layout.addWidget(calibration_group)
        main_layout.addWidget(ar_group)
        main_layout.addWidget(stereo_group)
        main_layout.addWidget(sift_group)

        # Set main widget and layout
        self.setCentralWidget(main_widget)

    def load_folder(self):
        # Open a dialog to select a folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        # If a folder was selected, process it
        if folder_path:
            self.image_paths.clear() # Clear previous images if needed
            
            # Supported image formats
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
            
            # Iterate through files in the selected folder
            for filename in os.listdir(folder_path):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(folder_path, filename)
                    self.image_paths.append(full_path)

    def load_imageL(self):
        # Open a file dialog to select an image file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)")

        if file_path:
            # Load the image with OpenCV
            self.imageL = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
    def load_imageR(self):
        # Open a file dialog to select an image file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)")

        if file_path:
            # Load the image with OpenCV
            self.imageR = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def find_corners(self):
        # Chessboard properties
        chessboard_size = (11, 8) # chessboard's width and height
        square_size = 0.02 # 0.02m unit per square
        
        # 3D object points (same for all images) [[0.   0.   0.  ][0.02 0.   0.  ][0.04 0.   0.  ]...[0.2  0.   0.  ][0.   0.02 0.  ]...[0.18 0.14 0.  ][0.2  0.14 0.  ]]
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # Prepare lists to store object points and image points for all images
        self.objectPoints = []
        self.imagePoints = []

        for image in self.image_paths:
            # Load the image in grayscale
            self.grayimg = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

            # Define chessboard size
            chessboard_size = (11, 8)

            # Detect chessboard corners
            ret, corners = cv2.findChessboardCorners(self.grayimg, chessboard_size)

            # Check if corners were found before proceeding
            if ret:
                # Define parameters for corner refinement
                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
                
                # Refine corner positions
                corners = cv2.cornerSubPix(self.grayimg, corners, winSize, zeroZone, criteria)

                # Append points for calibration
                self.objectPoints.append(objp)
                self.imagePoints.append(corners)
                
                # Draw the refined corners on the original image
                color_img = cv2.cvtColor(self.grayimg, cv2.COLOR_GRAY2BGR) # Convert grayscale to BGR for color overlay
                cv2.drawChessboardCorners(color_img, chessboard_size, corners, ret)
                
                # Resize the image to make it smaller
                scale_factor = 0.3
                smaller_img = cv2.resize(color_img, (0, 0), fx=scale_factor, fy=scale_factor)
                
                # Display the resized image
                cv2.imshow("Chessboard corners", smaller_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Chessboard corners not found.")

    def find_intrinsic(self):
        img_size = (2048, 2048) # image size
        # Perform camera calibration
        ret, self.ins, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objectPoints, self.imagePoints, img_size, None, None)

        # Output the calibration results if successful
        if ret:
            print("Intrinsic:")
            print(self.ins)

            dialog = MatrixDialog(self.ins, "Intrinsic")
            dialog.exec_()
        else:
            print("Calibration failed.")

    def find_extrinsic(self):
        # get QComboBox input image_number
        self.image_number = self.spin_box.value()-1
        # print(image_number+1)
        rvec = self.rvecs[self.image_number]
        tvec = self.tvecs[self.image_number]

        rotation_matrix = cv2.Rodrigues(rvec)[0]
        extrinsic_matrix = np.hstack((rotation_matrix , tvec))

        print("Extrinsic:")
        print(extrinsic_matrix)
        dialog = MatrixDialog(extrinsic_matrix, "Extrinsic")
        dialog.exec_()

    def find_distortion(self):
        print("Distortion:")
        print(self.dist)
        dialog = MatrixDialog(self.dist, "Distortion")
        dialog.exec_()

    def show_result(self):
        grayimg = cv2.imread(self.image_paths[self.image_number])
        result_img = cv2.undistort(grayimg, self.ins, self.dist)

        # Resize the image to make it smaller
        scale_factor = 0.4
        smaller_grayimg = cv2.resize(grayimg, (0, 0), fx=scale_factor, fy=scale_factor)
        smaller_result_img = cv2.resize(result_img, (0, 0), fx=scale_factor, fy=scale_factor)
        
        cv2.imshow("Distorted", smaller_grayimg)
        cv2.imshow("Undistorted", smaller_result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_char_points1(self, char):
        fs = cv2.FileStorage('../Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_onboard.txt', cv2.FILE_STORAGE_READ)
        char_points = fs.getNode(char).mat()
        fs.release()
        return char_points

    def show_words_board(self):
        # Chessboard properties
        chessboard_size = (11, 8) # chessboard's width and height
        square_size = 0.02 # 0.02m unit per square
        
        # 3D object points (same for all images) [[0.   0.   0.  ][0.02 0.   0.  ][0.04 0.   0.  ]...[0.2  0.   0.  ][0.   0.02 0.  ]...[0.18 0.14 0.  ][0.2  0.14 0.  ]]
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # Prepare lists to store object points and image points for all images
        objectPoints = []
        imagePoints = []

        for image in self.image_paths:
            # Load the image in grayscale
            grayimg = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

            # Define chessboard size
            chessboard_size = (11, 8)

            # Detect chessboard corners
            ret, corners = cv2.findChessboardCorners(grayimg, chessboard_size)

            # Check if corners were found before proceeding
            if ret:
                # Define parameters for corner refinement
                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
                
                # Refine corner positions
                corners = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)

                # Append points for calibration
                objectPoints.append(objp)
                imagePoints.append(corners)
            else:
                print("Chessboard corners not found.")

        img_size = (2048, 2048)  # image size
        # Perform camera calibration
        ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, img_size, None, None)

        # Get word from input
        word = self.text_edit.toPlainText().upper()
        # print(word)
        if len(word) > 6:
            print("Error: Word must be 6 characters or fewer.")
            return

        for i in range(1, 6):
            # Load the image
            img = cv2.imread(self.image_paths[i-1])

            for idx, char in enumerate(word):
                # Load 3D points for character
                char_points = self.load_char_points1(char)
                if char_points is None:
                    print(f"Error: Character '{char}' not found in database.")
                    continue
                
                # Translate character points to position it on the chessboard
                if idx == 0:
                    translation_offset = np.array([7, 5, 0])
                elif idx == 1:
                    translation_offset = np.array([4, 5, 0])
                elif idx == 2:
                    translation_offset = np.array([1, 5, 0])
                elif idx == 3:
                    translation_offset = np.array([7, 2, 0])
                elif idx == 4:
                    translation_offset = np.array([4, 2, 0])
                elif idx == 5:
                    translation_offset = np.array([1, 2, 0])
                char_points_translated = char_points + translation_offset

                # Ensure char_points_translated is in float32 format for cv2.projectPoints
                char_points_translated = char_points_translated.astype(np.float32) * 0.02 # Adjust spacing between letters
                # Ensure the shape is (N, 1, 3) for `cv2.projectPoints`
                char_points_translated = char_points_translated.reshape(-1, 1, 3)

                # Project points onto the image
                new_char_points, _ = cv2.projectPoints(char_points_translated, rvecs[i - 1], tvecs[i - 1], ins, dist)
                # Draw character lines on image
                for line in new_char_points.reshape(-1, 2, 2):
                    pt1 = tuple(line[0].ravel().astype(int))
                    pt2 = tuple(line[1].ravel().astype(int))
                    cv2.line(img, pt1, pt2, (250-idx*40, 50, 0+idx*40), 20) # B, G, R

            # Resize the image to make it smaller
            scale_factor = 0.3 
            smaller_grayimg = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            # Show the result for 1 second
            cv2.imshow(f"Image {i} - {word}", smaller_grayimg)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def load_char_points2(self, char):
        fs = cv2.FileStorage('../Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_vertical.txt', cv2.FILE_STORAGE_READ)
        char_points = fs.getNode(char).mat()
        fs.release()
        return char_points

    def show_words_vertical(self):
        # Chessboard properties
        chessboard_size = (11, 8) # chessboard's width and height
        square_size = 0.02 # 0.02m unit per square
        
        # 3D object points (same for all images) [[0.   0.   0.  ][0.02 0.   0.  ][0.04 0.   0.  ]...[0.2  0.   0.  ][0.   0.02 0.  ]...[0.18 0.14 0.  ][0.2  0.14 0.  ]]
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # Prepare lists to store object points and image points for all images
        objectPoints = []
        imagePoints = []

        for image in self.image_paths:
            # Load the image in grayscale
            grayimg = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

            # Define chessboard size
            chessboard_size = (11, 8)

            # Detect chessboard corners
            ret, corners = cv2.findChessboardCorners(grayimg, chessboard_size)

            # Check if corners were found before proceeding
            if ret:
                # Define parameters for corner refinement
                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
                
                # Refine corner positions
                corners = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)

                # Append points for calibration
                objectPoints.append(objp)
                imagePoints.append(corners)
            else:
                print("Chessboard corners not found.")

        img_size = (2048, 2048) # image size
        # Perform camera calibration
        ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, img_size, None, None)

        # Get word from input
        word = self.text_edit.toPlainText().upper()
        if len(word) > 6:
            print("Error: Word must be 6 characters or fewer.")
            return

        for i in range(1, 6):
            # Load the image
            img = cv2.imread(self.image_paths[i-1])

            for idx, char in enumerate(word):
                # Load 3D points for character
                char_points = self.load_char_points2(char)
                if char_points is None:
                    print(f"Error: Character '{char}' not found in database.")
                    continue
                
                # Translate character points to position it on the chessboard
                if idx == 0:
                    translation_offset = np.array([7, 5, 0])
                elif idx == 1:
                    translation_offset = np.array([4, 5, 0])
                elif idx == 2:
                    translation_offset = np.array([1, 5, 0])
                elif idx == 3:
                    translation_offset = np.array([7, 2, 0])
                elif idx == 4:
                    translation_offset = np.array([4, 2, 0])
                elif idx == 5:
                    translation_offset = np.array([1, 2, 0])
                char_points_translated = char_points + translation_offset

                # Ensure char_points_translated is in float32 format for cv2.projectPoints
                char_points_translated = char_points_translated.astype(np.float32) * 0.02 # Adjust spacing between letters
                # Ensure the shape is (N, 1, 3) for `cv2.projectPoints`
                char_points_translated = char_points_translated.reshape(-1, 1, 3)

                # Project points onto the image
                new_char_points, _ = cv2.projectPoints(char_points_translated, rvecs[i - 1], tvecs[i - 1], ins, dist)

                # Draw character lines on image
                for line in new_char_points.reshape(-1, 2, 2):
                    pt1 = tuple(line[0].ravel().astype(int))
                    pt2 = tuple(line[1].ravel().astype(int))
                    cv2.line(img, pt1, pt2, (250-idx*40, 0+idx*40, 50), 20) # B, G, R

            # Resize the image to make it smaller
            scale_factor = 0.3
            smaller_grayimg = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            # Show the result for 1 second
            cv2.imshow(f"Image {i} - {word}", smaller_grayimg)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def stereo_disparity(self):
        # Find the disparity map/image based on Left and Right stereo images
        # Use OpenCV StereoBM class to build StereoBM objects
        # numDisparities (int): The disparity search range must be positive and divisible by 16
        # blockSize (int): The size of blocks compared by the algorithm, must be odd and within the range [5, 51]
        stereo = cv2.StereoBM.create(numDisparities=432, blockSize=25)
        disparity = stereo.compute(self.imageL, self.imageR)

        # normalized to [0, 255]
        normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        normalized = np.uint8(normalized)

        # Resize the image to make it smaller
        scale_factor = 0.3
        smaller_imgL = cv2.resize(self.imageL, (0, 0), fx=scale_factor, fy=scale_factor)
        smaller_imgR = cv2.resize(self.imageR, (0, 0), fx=scale_factor, fy=scale_factor)
        smaller_img = cv2.resize(normalized, (0, 0), fx=scale_factor, fy=scale_factor)
        
        cv2.imshow("ImgL", smaller_imgL)
        cv2.imshow("ImgR", smaller_imgR)
        cv2.imshow("Disparity Map", smaller_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_image1(self):
        # Open a file dialog to select an image file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)")

        if file_path:
            # Load the image with OpenCV
            self.image1 = cv2.imread(file_path)

    def load_image2(self):
        # Open a file dialog to select an image file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)")

        if file_path:
            # Load the image with OpenCV
            self.image2 = cv2.imread(file_path)

    def sift_keypoints(self):
        # Convert image to grayscale image
        gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)

        # Use OpenCV SIFT detector to detect keypoints and descriptors.
        sift = cv2.SIFT_create() # Create a SIFT detector
        keypoints, descriptors = sift.detectAndCompute(gray, None) # Many SIFT keypoints, each keypoint has its descriptor
        
        # draw the keypoints
        img = cv2.drawKeypoints(gray, keypoints, None, color=(0,255,0))

        # Resize the image to make it smaller
        scale_factor = 0.2
        smaller_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        
        cv2.imshow("Keypoints", smaller_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def matched_keypoints(self):
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        sift1 = cv2.SIFT_create()
        keypoints1, descriptors1 = sift1.detectAndCompute(gray1, None)
        sift2 = cv2.SIFT_create()
        keypoints2, descriptors2 = sift2.detectAndCompute(gray2, None)

        # Find match keypoints of two images
        matches = cv2.BFMatcher().knnMatch(descriptors1, descriptors2, k=2)

        # Extract Good Matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        good_matches = np.expand_dims(good_matches, 1)

        # Draw the matched feature points between two image
        img = cv2.drawMatchesKnn(gray1, keypoints1, gray2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Resize the image to make it smaller
        scale_factor = 0.2
        smaller_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        
        cv2.imshow("Keypoints", smaller_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
# Initialize and run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
