clear all
name="Rectangular";

listing = dir(name)
listing=listing(3:97);
sp3i=[8:5:48];
sp3o=[53:5:93];
tr_healthy=[3]
tr_inner=[sp3i ];
tr_outer=[sp3o ];

inner_vibration=[];
inner_audio=[];
a=8

for i=1:1:length(tr_inner)
   
    load([listing(tr_inner(i)).folder '\' listing(tr_inner(i)).name]); 
   wn=floor(length(data)/4096);
   data=data(1:wn*4096,:);
   ir1=data(:,a);
   inner_vib=reshape(ir1,4096,length(ir1)/4096)';
   ir1=data(:,3);
   inner_aud=reshape(ir1,4096,length(ir1)/4096)';
   inner_vibration=[inner_vibration; inner_vib];
   inner_audio=[inner_audio; inner_aud];

end



outer_vibration=[];
outer_audio=[];

for i=1:1:length(tr_outer)
   
    load([listing(tr_outer(i)).folder '\' listing(tr_outer(i)).name]); 
   wn=floor(length(data)/4096);
   data=data(1:wn*4096,:);
   ir1=data(:,a);
   inner_vib=reshape(ir1,4096,length(ir1)/4096)';
   ir1=data(:,3);
   inner_aud=reshape(ir1,4096,length(ir1)/4096)';
   outer_vibration=[outer_vibration; inner_vib];
   outer_audio=[outer_audio; inner_aud];

end



healthy_vibration=[];
healthy_audio=[];

for i=1:1:length(tr_healthy)
   
    load([listing(tr_healthy(i)).folder '\' listing(tr_healthy(i)).name]); 
   wn=floor(length(data)/4096);
   data=data(1:wn*4096,:);
   ir1=data(:,a);
   inner_vib=reshape(ir1,4096,length(ir1)/4096)';
   ir1=data(:,3);
   inner_aud=reshape(ir1,4096,length(ir1)/4096)';
   healthy_vibration=[healthy_vibration; inner_vib];
   healthy_audio=[healthy_audio; inner_aud];

end







h1v=healthy_vibration(1:124,:);
h2v=healthy_vibration(125:end,:);

i1v=inner_vibration(1:62,:);
i2v=inner_vibration(63:end,:);

o1v=outer_vibration(1:62,:);
o2v=outer_vibration(63:end,:);

h1a=healthy_audio(1:124,:);
h2a=healthy_audio(125:end,:);

i1a=inner_audio(1:62,:);
i2a=inner_audio(63:end,:);

o1a=outer_audio(1:62,:);
o2a=outer_audio(63:end,:);

label=-1*ones(size(h1a,1)+size(o1a,1)+size(i1a,1),2);
label(1:size(h1a,1),1)=-1*label(1:size(h1a,1),1);
label(size(h1a,1):size(h1a,1)+size(o1a,1),2)=-1*label(size(h1a,1):size(h1a,1)+size(o1a,1),2);
label(size(h1a,1)+size(o1a,1):size(h1a,1)+size(o1a,1)+size(i1a,1),2)=-1*label(size(h1a,1)+size(o1a,1):size(h1a,1)+size(o1a,1)+size(i1a,1),2);

clear_sig=[h1v;o1v;i1v];
noisy_sig=[h1a;o1a;i1a];
label_sig=label;
bb=[clear_sig noisy_sig label_sig];
random_x = bb(randperm(size(bb, 1)), :);

vib_sig=random_x(:,1:4096);
audio_sig=random_x(:,4097:8192);
label_sig=random_x(:,8193:end);



save('MachineB/S3/train/audio_sig.mat','audio_sig')
save('MachineB/S3/train/vib_sig.mat','vib_sig')
save('MachineB/S3/train/label_sig.mat','label_sig')




label=-1*ones(size(h2a,1)+size(o2a,1)+size(i2a,1),2);
label(1:size(h2a,1),1)=-1*label(1:size(h2a,1),1);
label(size(h2a,1):size(h2a,1)+size(o2a,1),2)=-1*label(size(h2a,1):size(h2a,1)+size(o2a,1),2);
label(size(h2a,1)+size(o2a,1):size(h2a,1)+size(o2a,1)+size(i2a,1),2)=-1*label(size(h2a,1)+size(o2a,1):size(h2a,1)+size(o2a,1)+size(i2a,1),2);

vib_sig=[h2v;o2v;i2v];
audio_sig=[h2a;o2a;i2a];
label_sig=label;



save('MachineB/S3/test/audio_sig.mat','audio_sig')
save('MachineB/S3/test/vib_sig.mat','vib_sig')
save('MachineB/S3/test/label_sig.mat','label_sig')


