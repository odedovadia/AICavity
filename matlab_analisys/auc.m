function res = auc(prediction,labels)
%ROC Summary of this function goes here
%   Detailed explanation goes here
nt = sum(labels);
tot_el = numel(labels);
samples = 10000;
accuracy = @(t)((sum((prediction<=t ).*double(labels==0))/(tot_el-nt)+sum((prediction>t ).*double(labels))/nt)/2);
sample_auc= linspace(0,1,samples);
dt = 1/samples;
res= sum(accuracy(sample_auc).*dt);
end

