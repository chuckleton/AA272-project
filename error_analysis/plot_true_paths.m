lla_true = readcell('true_path_coords/lake_coords.csv');
lla_true = cell2mat(lla_true);

subplot(1,3,1)
geoplot(lla_true(:,1),lla_true(:,2),'w-','LineWidth',2)
geobasemap satellite

lla_true = readcell('true_path_coords/track_coords.csv');
lla_true = cell2mat(lla_true);

subplot(1,3,2)
geoplot(lla_true(:,1),lla_true(:,2),'w-','LineWidth',2)
geobasemap satellite

lla_true = readcell('true_path_coords/random_coords.csv');
lla_true = cell2mat(lla_true);

subplot(1,3,3)
geoplot(lla_true(:,1),lla_true(:,2),'w-','LineWidth',2)
geobasemap satellite

set(gcf,'Position',[0 0 1280 600])
exportgraphics(gcf,'paths.png','Resolution',300)

%%

clc; clear; close all;

lla_true = readcell('true_path_coords/lake_coords.csv');
lla_true = cell2mat(lla_true);

geoplot(lla_true(:,1),lla_true(:,2),'-','LineWidth',5)
hold on
geobasemap satellite

lla_estimate = readcell('../WLS_LLA/WLS_LLA_lake.csv');
lla_estimate = cell2mat(lla_estimate(2:end-8,2:3));
lla_estimate = [lla_estimate zeros(size(lla_estimate,1),1)];
geoplot(lla_estimate(:,1),lla_estimate(:,2),'.r','MarkerSize',10)
set(gcf,'Position',[0 0 1920 1080])
legend('True Path','WLS Estimate')

exportgraphics(gcf,'true_with_estimate.png','Resolution',300)
