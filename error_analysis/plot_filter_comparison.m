clc; clear; close all;

random_initial = [37.4308352 -122.1645004];
random_coords = readcell('true_path_coords/random_coords.csv');
random_coords = random_coords(1:500,:);
random_wls = readcell('../WLS_LLA/WLS_LLA_rand_path.csv');
random_wls = cell2mat(random_wls(2:1000,2:3));
random_wls = [random_wls zeros(size(random_wls,1),1)];
random_wls_f_1 = readcell('../WLS_LLA/WLS_LLA_rand_path_xy_filtered_0.00028333333333333335_10.0_1.5.csv');
random_wls_f_1 = convert_to_lla(random_wls_f_1,random_initial);
random_wls_f_2 = readcell('../WLS_LLA/WLS_LLA_rand_path_xy_filtered_0.005_10.0_1.5.csv');
random_wls_f_2 = convert_to_lla(random_wls_f_2,random_initial);
random_wls_f_3 = readcell('../WLS_LLA/WLS_LLA_rand_path_xy_filtered_0.0125_10.0_1.5.csv');
random_wls_f_3 = convert_to_lla(random_wls_f_3,random_initial);

datasets = {random_coords, random_wls, 'WLS';
    random_coords, random_wls_f_1, '1';
%     random_coords, random_wls_f_2, '2';
%     random_coords, random_wls_f_3, '3'
    };
num_datasets = size(datasets,1);

lla_true = random_coords;
lla_true = cell2mat(lla_true(:,1:2));
lla_true = [lla_true zeros(size(lla_true,1),1)];
geoplot(lla_true(:,1),lla_true(:,2),'.-','MarkerSize',15,'LineWidth',2)
hold on

for j = 1:num_datasets

    lla_estimate = datasets{j,2};
    % Convert true path LLA coordinates to flat earth coordinates

    flat_true = lla2flat(lla_true,lla_true(1,1:2),90,0);
    flat_true(:,2) = -flat_true(:,2);
    
    if j == 2
        geoplot(lla_estimate(:,1),lla_estimate(:,2),'.--','MarkerSize',15,'LineWidth',2)
    else
    % Plot position estimates
    geoplot(lla_estimate(:,1),lla_estimate(:,2),'.','MarkerSize',15)
    end
end

legend('True Path','WLS','Filtered')
set(gcf,'Position',[0 0 500 900])
title("P = 0.000283, M = 10.0, F = 1.5")
exportgraphics(gcf,'filteroverlay.png')