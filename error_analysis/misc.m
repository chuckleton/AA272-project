%% Storage for misc code snippets

%     frames(upto) = struct('cdata',[],'colormap',[]);

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

%         subplot(1,2,2)
%         plot(errors(1:i),'bl')
%         hold on
%         plot(avg_error_mags(1:i),'r')
%         grid on
%         xlabel('Timestep')
%         ylabel('Error (m)')
%         legend('Error','Average Error')


        figure(map)
        subplot(2,num_datasets,j)
        geoplot(lla_estimate(i,1),lla_estimate(i,2),'.r','MarkerSize',20)
        geoplot(lla_coords(1:2,1),lla_coords(1:2,2),'w','LineWidth',2)
        geolimits([lla_coords(4,1) lla_coords(3,1)], [lla_coords(4,2) lla_coords(3,2)])

        drawnow
        frames(i) = getframe(gcf);

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