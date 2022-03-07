%% Calculate Errors
% Calculates total distance error of estimated position solutions by
% finding perpendicular distance to true path

clc; clear; close all;

% Uncomment for random
lla_true = readcell('true_path_coords/random_coords.csv');
lla_estimate = readcell('../WLS_LLA/WLS_LLA_rand_path.csv');
upto = 2002;

% Uncomment for lake
% lla_true = readcell('true_path_coords/lake_coords.csv');
% lla_estimate = readcell('../WLS_LLA/WLS_LLA_lake.csv');
% upto = 1465;

% Uncomment for track
% lla_true = readcell('true_path_coords/track_coords.csv');
% lla_estimate = readcell('../WLS_LLA/WLS_LLA_track.csv');
% lla_estimate = lla_estimate(50:end-100,:);
% upto = size(lla_estimate,1);

true_start = 1;
true_end = size(lla_true,1);

lla_true = cell2mat(lla_true(true_start:true_end,1:2));
lla_estimate = cell2mat(lla_estimate(1:upto,2:4));

% Convert true path LLA coordinates to flat earth coordinates
lla_true = [lla_true zeros(size(lla_true,1),1)];
flat_true = lla2flat(lla_true,lla_true(1,1:2),90,0);
flat_true(:,2) = -flat_true(:,2);

% Plot true path
flat = figure;
figure(flat)
set(gcf,'Position',[0 0 1920 1080])
subplot(1,2,1)
geoplot(lla_true(:,1),lla_true(:,2),'.-','MarkerSize',20,'LineWidth',2)
% labels = num2cell(true_start:true_end);
% text(lla_true(:,1),lla_true(:,2),labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
geobasemap satellite
hold on

% Plot position estimates
geoplot(lla_estimate(:,1),lla_estimate(:,2),'.r','MarkerSize',20)

% Convert estimated path LLA coordinates to flat earth coordinates, using
% first true path coordinate as reference
flat_estimate = lla2flat(lla_estimate,lla_true(1,1:2),90,0);
flat_estimate(:,2) = -flat_estimate(:,2);

% Calculate errors
num_points = size(flat_estimate,1);
errors = zeros(num_points,1);
avg_error_mags = errors;

pt1_prevs = repmat(flat_estimate(1,1:2),10,1);

frames(upto) = struct('cdata',[],'colormap',[]);
true_coords = zeros(upto,2);

for i = 1:upto
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

    zoom = 500;
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

    %     figure(flat)
    %     subplot(1,2,1)
    %     geoplot(lla_estimate(i,1),lla_estimate(i,2),'.r','MarkerSize',20)
    %     geoplot(lla_coords(1:2,1),lla_coords(1:2,2),'w','LineWidth',2)
    %     geolimits([lla_coords(4,1) lla_coords(3,1)], [lla_coords(4,2) lla_coords(3,2)])
    %     subplot(1,2,2)
    %     plot(errors(1:i),'bl')
    %     hold on
    %     plot(avg_error_mags(1:i),'r')
    %     grid on
    %     xlabel('Timestep')
    %     ylabel('Error (m)')
    %     legend('Error','Average Error')
    %     drawnow
    %     frames(i) = getframe(gcf);
end

subplot(1,2,2)
plot(errors,'bl')
hold on
plot(avg_error_mags,'r')
grid on
xlabel('Timestep')
ylabel('Error (m)')
legend('Error','Average Error')

delta_true_coords = diff(true_coords);
dist_true_coords = sqrt(delta_true_coords(:,1).^2 + delta_true_coords(:,2).^2);
total_dist = sum(dist_true_coords);

% flat.Visible = 'on';
%
% v = VideoWriter('lake','MPEG-4');
% v.Quality = 100;
% v.FrameRate = 10;
% open(v)
% writeVideo(v,frames)
% close(v)

% grid on

% xlabel('X (m)')
% ylabel('Y (m)')

% figure()
% plot(errors)
% xlabel('Timestep')
% ylabel('Error (m)')

disp(['Total error: ',num2str(sum(abs(errors))),' m'])
disp(['Total distance: ',num2str(total_dist),' m'])
disp(['Overall error rate: ',num2str(sum(abs(errors))/total_dist),' meters per meter run'])