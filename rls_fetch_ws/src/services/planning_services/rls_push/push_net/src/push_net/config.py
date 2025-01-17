import os

batch_size = 25 # should match number of sampled actions

arch = {
        'simcom':'cnn4_16163232_reg_sim100_com1000_all', # POLY + ELLP: sim + com
        'sim': 'cnn4_16163232_reg_sim100_com1000_all_sim', # POLY + ELLP: sim only
        'nomem': 'cnn4_16163232_reg_sim100_com1000_all_sim_nomem' # no history
       }

label_path = 'full_dataset.json'

model_path = 'model'
#best_model_name = 'model_best_' + arch + '.pth.tar'
#checkpoint_name = 'checkpoint_' + arch + '.pth.tar'

#best_model_path = os.path.join(model_path, best_model_name)
#checkpoint_path = os.path.join(model_path, checkpoint_name)


