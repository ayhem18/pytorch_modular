"""
This is a quick test for the KNN algorithm implementation
"""

import torch, os, shutil
import torchvision.transforms as tr
import numpy as np

from torchvision.datasets import FashionMNIST

from mypt.subroutines.neighbors.knn import KNN, KnnClassifier
from mypt.code_utilities import directories_and_files as dirf


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def knn_test_1(): 
    
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    dirf.process_path(data_dir, file_ok=False)
    
    # this object will download the 
    img_transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

    val_ds = FashionMNIST(root=data_dir, train=False, transform=img_transform, download=True)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_features=32 * 32, out_features=128))

    knn_classifier = KNN(train_ds=val_ds,
                        model=model, 
                        train_ds_inference_batch_size=1000, 
                        process_sample_ds=lambda x:x[0], # dataset[i] returns a tuple: the image and the label
                        process_model_output=None,
                        model_ckpnt=None)


    for m in ['cosine_sim', 'euclidean']:            
        values_res, indices_res = knn_classifier.predict(val_ds=val_ds, 
                                            inference_batch_size=1000, 
                                            num_neighbors=11, 
                                            process_sample_ds=lambda x: x[0],
                                            measure=m,
                                            measure_as_similarity= (m == 'cosine_sim')
                                            )

        device = knn_classifier.inference_device
        
        msr = knn_classifier._measures(m)
        if isinstance(msr, torch.nn.Module):
            msr = msr.to(device)

        for sample_index, dis in enumerate(values_res):
            # process the validation sample
            val_sample = val_ds[sample_index][0].to(device)
            neighbors = torch.stack([knn_classifier.process_sample_ds(knn_classifier.train_ds[i]) for i in indices_res[sample_index, :]], dim=0).to(device)
            
            dis = np.expand_dims(dis, axis=0)
        
            # pass through the model
            knn_classifier._load_model()

            with torch.no_grad():
                neighbors = knn_classifier.process_model_output(knn_classifier.model, neighbors)
                val_sample = knn_classifier.process_model_output(knn_classifier.model, val_sample)
                dis_true = msr(val_sample, neighbors)
                
                atol=10 ** -4 if m == 'euclidean' else 10 ** -8 # the distances are very close to zero and the numerical precesion is much lower...

            assert torch.allclose(dis_true.cpu(), torch.from_numpy(dis), atol=atol), "The precomputed distances do not match the actual ones !!!"

    # remove the extra repositories
    shutil.rmtree(data_dir)

def knn_test_2(): 
    
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    dirf.process_path(data_dir, file_ok=False)
    
    img_transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

    # this object will download dowload the datasert
    val_ds = FashionMNIST(root=data_dir, train=False, transform=img_transform, download=True)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_features=32 * 32, out_features=128))

    knn_classifier = KnnClassifier(train_ds=val_ds,
                        model=model, 
                        train_ds_inference_batch_size=1000, 
                        process_sample_ds=lambda x:x[0], # dataset[i] returns a tuple: the image and the label
                        process_model_output=None,
                        model_ckpnt=None)


    for m in ['cosine_sim', 'euclidean']:            
        label_predictions = knn_classifier.predict(val_ds=val_ds, 
                                        inference_batch_size=1000, 
                                        num_neighbors=11, 
                                        process_sample_ds=lambda x: x[0],
                                        measure=m,
                                        measure_as_similarity= (m == 'cosine_sim')
                                        )

        assert np.all(np.isin(label_predictions, list(range(10)))), "All predictions should be integers from 0 to 9"


    # remove the extra repositories
    shutil.rmtree(data_dir)


if __name__ == '__main__':
    knn_test_1()
    knn_test_2()
