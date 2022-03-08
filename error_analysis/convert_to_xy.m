clc; clear; close all;

lla_estimate = readcell('../WLS_LLA/WLS_LLA_lake.csv');
lla_estimate = cell2mat(lla_estimate(2:1614,:));
flat_estimate = lla2flat(lla_estimate(:,2:4),lla_estimate(1,2:3),90,0);
flat_estimate(:,2) = -flat_estimate(:,2);
flat_estimate = [lla_estimate(:,1) flat_estimate];

plot(flat_estimate(:,2),flat_estimate(:,3));
writematrix(flat_estimate,'WLS_LLA_lake_xy.csv')
axis equal