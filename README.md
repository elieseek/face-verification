# face-verification
A face recognition model trained to create unique embeddings using Generalised end-to-end loss.

## Completed
- Dataset restructuring
- CNN embedding model
- [GE2E loss](https://arxiv.org/abs/1710.10467) for embeddings
- Training loop

## To-do
- [ ] Calculate EER
- [ ] train networks + create new
- [ ] dockerise trained model
- [ ] deploy demo

# Training the model
- Download and extract the cropped & aligned [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) along with identity annotations and partition text files.
- Edit config.py to point to the images folder and text files
- Edit config.py to point to an existing model and checkpoint folder
- Once sure that your folders and config are set up correctly, run dataset.py to restructure the folders
- Run train.py :)
