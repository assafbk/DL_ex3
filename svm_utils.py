import torch

def sample_balanced_data(num_of_samples_per_class, num_of_classes, data_loader, im_h, im_w):

    total_num_of_samples = num_of_samples_per_class*num_of_classes
    samples = torch.zeros([total_num_of_samples, im_h, im_w])
    labels = torch.zeros([total_num_of_samples,1])
    per_class_counter = torch.zeros([num_of_classes,1])
    i=0
    for x,y in data_loader:
        for k in range(x.shape[0]):
            if per_class_counter[y[k]] < num_of_samples_per_class:
                samples[i,:,:] = x[k,:,:]
                labels[i] = y[k]
                per_class_counter[y[k]] += 1
                i+=1

            if torch.all(per_class_counter == num_of_samples_per_class):
                return samples, torch.squeeze(labels)



