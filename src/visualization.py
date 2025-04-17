import random
import matplotlib.pyplot as plt

# Pick a few samples from validation set
sample_gen = SliceDataset(val_img, val_mask, batch_size=1, is_train=False)

def visualize_predictions(model, dataset, num_samples=5):
    plt.figure(figsize=(15, num_samples * 3))
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        img_batch, mask_batch = dataset[idx]
        pred_mask = model.predict(img_batch)[0, :, :, 0]
        true_mask = mask_batch[0, :, :, 0]
        img = img_batch[0, :, :, 0]

        # Plot
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img, cmap='gray')
        plt.title("Input Slice")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "segmentation_visualization.png"))
    plt.show()

visualize_predictions(model, sample_gen)
