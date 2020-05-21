# face-verification
A face recognition model trained to create unique embeddings using generalised end-to-end loss.

## Completed
- Dataset restructuring
- CNN embedding model
- [GE2E loss](https://arxiv.org/abs/1710.10467) for embeddings
- Training loop
- Z normalizing input
- Trained network
- Flask API

## To-do
- [ ] Fix package
- [ ] Calculate EER
- [ ] Deploy demo

# Training the model
1. Download and extract the cropped & aligned [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) along with identity annotations and partition text files.
2. Edit config.py to point to the images folder and text files
3. Edit config.py to point to an existing model and checkpoint folder
4. Once sure that your folders and config are set up correctly, run dataset.py to restructure the folders
5. Run train_net.py :)

# Requirements
- numpy
- opencv-python
- torch