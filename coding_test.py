import cv2
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

class PuzzleSolver:
    def __init__(self, original_image_path, pieces_data_path, piece_width=32, piece_height=27):
        self.original_image = cv2.imread(original_image_path)
        self.piece_width = piece_width
        self.piece_height = piece_height

        with open(pieces_data_path, 'r') as file:
            self.puzzle_pieces_data = json.load(file)

    def process_piece(self, piece_info):
        piece_id = piece_info["id"]
        piece_image = cv2.imread(f'puzzle_pieces/{piece_id}.jpg')

        if piece_image is None:
            print(f"Failed to load image for piece {piece_id}")
            return None

        result = cv2.matchTemplate(self.original_image, piece_image, cv2.TM_CCORR_NORMED) # most propriate
        
        _, _, min_loc, max_loc = cv2.minMaxLoc(result)

        # Using the coordinates of the maximum value to place the puzzle piece
        top_left = list(max_loc)
        
        rmd = top_left[0] % self.piece_width
        if rmd > (self.piece_width / 2):
            top_left[0] += self.piece_width - rmd
        else:
            top_left[0] -= rmd
        rmd = top_left[1] % self.piece_height
        if rmd > (self.piece_height / 2):
            top_left[1] += self.piece_height - rmd
        else:
            top_left[1] -= rmd
        
        bottom_right = (top_left[0] + self.piece_width, top_left[1] + self.piece_height)
        # Returning the processing result
        return (top_left, bottom_right, piece_image)

    def reconstruct_puzzle(self):
        reconstructed_puzzle = np.zeros_like(self.original_image)

        with Pool(cpu_count()) as p:
            results = list(tqdm(p.imap(self.process_piece, self.puzzle_pieces_data), total=len(self.puzzle_pieces_data), desc="Reconstructing puzzle"))

        for result in results:
            if result is not None:
                top_left, bottom_right, piece_image = result
                reconstructed_puzzle[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = piece_image
        return reconstructed_puzzle

if __name__ == "__main__":
    solver = PuzzleSolver('image.jpg', 'puzzle_pieces.json')
    reconstructed_puzzle = solver.reconstruct_puzzle()

    # Save the restored puzzle image
    cv2.imwrite('reconstructed_puzzle.jpg', reconstructed_puzzle)
