import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class MNISTCSVDrawer {

    private static final int IMAGE_SIZE = 28; // MNIST images are 28x28 pixels

    public static void main(String[] args) {
        String csvFilePath = "mnist_train_100.csv"; // Update with your CSV file path
        int imageIndex = 45; // The index of the image to display (0 for the first image)

        try {
            int[][] image = loadMNISTImageFromCSV(csvFilePath, imageIndex);
            if (image != null) {
                SwingUtilities.invokeLater(() -> createAndShowGUI(image));
            } else {
                System.out.println("Failed to load the image.");
            }
        } catch (IOException e) {
            System.err.println("Error reading the CSV file: " + e.getMessage());
        }
    }

    private static int[][] loadMNISTImageFromCSV(String csvFilePath, int imageIndex) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            List<int[]> images = new ArrayList<>();

            while ((line = br.readLine()) != null) {
                String[] pixels = line.split(",");
                int[] image = new int[pixels.length];
                for (int i = 0; i < pixels.length; i++) {
                    image[i] = Integer.parseInt(pixels[i]); // Parse pixel values
                }
                images.add(image);
            }

            if (imageIndex < images.size()) {
                // Convert 1D array to a 2D array (28x28)
                int[] flatImage = images.get(imageIndex);
                int[][] reshapedImage = new int[IMAGE_SIZE][IMAGE_SIZE];
                for (int row = 0; row < IMAGE_SIZE; row++) {
                    System.arraycopy(flatImage, row * IMAGE_SIZE, reshapedImage[row], 0, IMAGE_SIZE);
                }
                return reshapedImage;
            } else {
                System.err.println("Image index out of range.");
                return null;
            }
        }
    }

    private static void createAndShowGUI(int[][] image) {
        JFrame frame = new JFrame("MNIST Image Viewer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ImagePanel(image));
        frame.pack();
        frame.setVisible(true);
    }

    static class ImagePanel extends JPanel {
        private final int[][] image;

        public ImagePanel(int[][] image) {
            this.image = image;
            setPreferredSize(new Dimension(IMAGE_SIZE, IMAGE_SIZE));
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;

            for (int y = 0; y < IMAGE_SIZE; y++) {
                for (int x = 0; x < IMAGE_SIZE; x++) {
                    int pixelValue = image[y][x]; // Pixel value (0-255)
                    g2d.setColor(new Color(pixelValue, pixelValue, pixelValue)); // Grayscale color
                    g2d.fillRect(x, y, 1, 1); // Draw a single pixel
                }
            }
        }
    }
}
