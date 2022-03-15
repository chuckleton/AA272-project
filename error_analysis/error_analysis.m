%% Calculate Errors
% Calculates and plots total distance error of estimated position solutions by
% finding perpendicular distance to true path

clc; clear; close all;

% Import true paths
lake_coords = readcell('true_path_coords/lake_coords.csv');
lake_coords = lake_coords(150:end,:);
random_coords = readcell('true_path_coords/random_coords.csv');
random_coords = random_coords(1:500,:);
track_coords = readcell('true_path_coords/track_coords.csv');

% Import position estimates
lake_initial = [37.42050384	-122.1756194];
lake_wls = readcell('../WLS_LLA/WLS_LLA_lake.csv');
lake_wls = cell2mat(lake_wls(2:1000,2:3));
lake_wls = [lake_wls zeros(size(lake_wls,1),1)];
lake_wls_f_1 = readcell('../WLS_LLA/WLS_LLA_lake_xy_filtered_0.00028333333333333335_10.0_1.5.csv');
lake_wls_f_1 = convert_to_lla(lake_wls_f_1,lake_initial);
lake_wls_f_2 = readcell('../WLS_LLA/WLS_LLA_lake_xy_filtered_0.005_10.0_1.5.csv');
lake_wls_f_2 = convert_to_lla(lake_wls_f_2,lake_initial);
lake_wls_f_3 = readcell('../WLS_LLA/WLS_LLA_lake_xy_filtered_0.0125_10.0_1.5.csv');
lake_wls_f_3 = convert_to_lla(lake_wls_f_3,lake_initial);

random_initial = [37.4308352 -122.1645004];
random_wls = readcell('../WLS_LLA/WLS_LLA_rand_path.csv');
random_wls = cell2mat(random_wls(2:1000,2:3));
random_wls = [random_wls zeros(size(random_wls,1),1)];
random_wls_f_1 = readcell('../WLS_LLA/WLS_LLA_rand_path_xy_filtered_0.00028333333333333335_10.0_1.5.csv');
random_wls_f_1 = convert_to_lla(random_wls_f_1,random_initial);
random_wls_f_2 = readcell('../WLS_LLA/WLS_LLA_rand_path_xy_filtered_0.005_10.0_1.5.csv');
random_wls_f_2 = convert_to_lla(random_wls_f_2,random_initial);
random_wls_f_3 = readcell('../WLS_LLA/WLS_LLA_rand_path_xy_filtered_0.0125_10.0_1.5.csv');
random_wls_f_3 = convert_to_lla(random_wls_f_3,random_initial);

track_initial = [37.43112392 -122.1626866];
track_wls = readcell('../WLS_LLA/WLS_LLA_track.csv');
track_wls = cell2mat(track_wls(2:end,2:3));
track_wls = [track_wls zeros(size(track_wls,1),1)];
track_wls_f_1 = readcell('../WLS_LLA/WLS_LLA_track_xy_filtered_0.00028333333333333335_10.0_1.5.csv');
track_wls_f_1 = convert_to_lla(track_wls_f_1,track_initial);
track_wls_f_2 = readcell('../WLS_LLA/WLS_LLA_track_xy_filtered_0.005_10.0_1.5.csv');
track_wls_f_2 = convert_to_lla(track_wls_f_2,track_initial);
track_wls_f_3 = readcell('../WLS_LLA/WLS_LLA_track_xy_filtered_0.0125_10.0_1.5.csv');
track_wls_f_3 = convert_to_lla(track_wls_f_3,track_initial);

% Specify datasets to process
datasets = {lake_coords, lake_wls, 'WLS';
            lake_coords, lake_wls_f_1, 'P = 0.000283';
            lake_coords, lake_wls_f_2, 'P = 0.005';
            lake_coords, lake_wls_f_3, 'P = 0.0125'};
% datasets = {random_coords, random_wls, 'WLS';
%                 random_coords, random_wls_f_1, 'P = 0.000283';
%                 random_coords, random_wls_f_2, 'P = 0.005';
%                 random_coords, random_wls_f_3, 'P = 0.0125'};
% datasets = {track_coords, track_wls, 'WLS';
%             track_coords, track_wls_f_1, 'P = 0.000283';
%             track_coords, track_wls_f_2, 'P = 0.005';
%             track_coords, track_wls_f_3, 'P = 0.0125'};

num_datasets = size(datasets,1);

map = figure();
set(gcf,'Position',[0 0 1820 800])

for j = 1:num_datasets
    lla_true = datasets{j,1};
    lla_estimate = datasets{j,2};

%     lla_true = lla_true(1:200,:);
%     lla_estimate = lla_estimate(100:200,:);

    % Convert true path LLA coordinates to flat earth coordinates
    lla_true = cell2mat(lla_true(:,1:2));
    lla_true = [lla_true zeros(size(lla_true,1),1)];
    flat_true = lla2flat(lla_true,lla_true(1,1:2),90,0);
    flat_true(:,2) = -flat_true(:,2);

    % Plot true path
    subplot(2,num_datasets,j)
    geoplot(lla_true(:,1),lla_true(:,2),'.-','MarkerSize',10,'LineWidth',2)
    geobasemap satellite
    title(datasets{j,3})
    hold on

    % Plot position estimates
    geoplot(lla_estimate(:,1),lla_estimate(:,2),'.r','MarkerSize',10)

    % Convert estimated path LLA coordinates to flat earth coordinates, using
    % first true path coordinate as reference
    flat_estimate = lla2flat(lla_estimate,lla_true(1,1:2),90,0);
    flat_estimate(:,2) = -flat_estimate(:,2);

    % Calculate errors
    num_points = size(flat_estimate,1);
    errors = zeros(num_points,1);
    avg_error_mags = errors;

    pt1_prevs = repmat(flat_estimate(1,1:2),10,1);

    true_coords = zeros(size(lla_true,1),2);

    for i = 1:size(flat_estimate,1)
        coords = flat_estimate(i,1:2);
        % Calculate straight line distance from current coordinate to all true
        % path points
        deltas = flat_true(:,1:2) - repmat(coords,size(flat_true,1),1);
        distances = sum(deltas.^2,2).^0.5;
        % Find closest point
        [M,I] = min(distances);
        pt1 = flat_true(I,1:2);
        pt_vec = coords - pt1;

        if I < size(flat_true,1) && I > 1
            pt2 = flat_true(I+1,1:2);
            pt3 = flat_true(I-1,1:2);

            before_vec = pt3 - pt1;
            n_before = [-before_vec(2),before_vec(1)];
            after_vec = pt2 - pt1;
            n_after = [-after_vec(2),after_vec(1)];

            proj_before = dot(pt_vec,n_before)/norm(n_before);
            proj_after = dot(pt_vec,n_after)/norm(n_after);
            [errors(i),J] = min(abs([proj_before proj_after]));

            if J == 1
                proj_vec = n_before/norm(n_before);
                errors(i) = proj_before;
            else
                proj_vec = n_after/norm(n_after);
                errors(i) = proj_after;
            end

        elseif I == size(flat_true,1)
            pt3 = flat_true(I-1,1:2);
            before_vec = pt3 - pt1;
            n_before = [-before_vec(2),before_vec(1)];

            proj_before = dot(pt_vec,n_before)/norm(n_before);
            errors(i) = proj_before;
            proj_vec = n_before/norm(n_before);

        else
            pt2 = flat_true(I+1,1:2);
            after_vec = pt2 - pt1;
            n_after = [-after_vec(2),after_vec(1)];

            proj_after = dot(pt_vec,n_after)/norm(n_after);
            errors(i) = proj_after;
            proj_vec = n_after/norm(n_after);
        end

        avg_error_mags(i) = mean(abs(errors(1:i)));

        coords1 = [coords; coords - errors(i)*proj_vec];

        true_coord = coords - errors(i)*proj_vec;
        true_coords(i,:) = true_coord;

        zoom = 200;
        center = mean([pt1; pt1_prevs]);
        max_x = center(1) + zoom;
        min_x = center(1) - zoom;
        max_y = center(2) + zoom;
        min_y = center(2) - zoom;
        pt1_prevs = circshift(pt1_prevs,-1);
        pt1_prevs(end,:) = pt1;

        coords_to_plot = [coords1 [0;0];
            max_x max_y 0;
            min_x min_y 0];

        coords_to_plot(:,2) = -coords_to_plot(:,2);

        lla_coords = flat2lla(coords_to_plot,lla_true(1,1:2),90,0);

    end

    subplot(2,num_datasets,j+num_datasets)
    plot(errors,'bl')
    hold on
    plot(avg_error_mags,'r')
    grid on
    xlabel('Timestep')
    ylabel('Error (m)')
    legend('Error','Average Error')
    axis([0 length(errors) -30 30])

    delta_true_coords = diff(true_coords);
    dist_true_coords = sqrt(delta_true_coords(:,1).^2 + delta_true_coords(:,2).^2);
    total_dist = sum(dist_true_coords);

    disp(['Coefficients: ',datasets{j,3}])
    disp(['Total error: ',num2str(round(sum(abs(errors)))),' m'])
    disp(['Total distance: ',num2str(round(total_dist)),' m'])
    disp(['Overall error rate: ',num2str(round(sum(abs(errors))/total_dist,3)),' meters per meter run'])
    disp(' ')
end

exportgraphics(gcf,'lake.png','Resolution',300)