for i=1:1:5

filename_train="C:/Users/ozerc/Desktop/SoundvsVib/MachineB/S"+int2str(i)+"/train/"
filename_test="C:/Users/ozerc/Desktop/SoundvsVib/MachineB/S"+int2str(i)+"/test/"

load(filename_train+"audio_sig")
load(filename_train+"vib_sig")
load(filename_train+"label_sig")


ch1array=[vib_sig];
ch2array=[audio_sig];
labelarray=[label_sig];


save(['mbdata/ch1data_train_' int2str(i) '.mat'],'ch1array')
save(['mbdata/ch2data_train_' int2str(i) '.mat'],'ch2array')
save(['mbdata/labels_train_' int2str(i) '.mat'],'labelarray')

load(filename_test+"audio_sig")
load(filename_test+"vib_sig")
load(filename_test+"label_sig")


ch1array=[vib_sig];
ch2array=[audio_sig];
labelarray=[label_sig];


save(['mbdata/ch1data_test_' int2str(i) '.mat'],'ch1array')
save(['mbdata/ch2data_test_' int2str(i) '.mat'],'ch2array')
save(['mbdata/labels_test_' int2str(i) '.mat'],'labelarray')

end
