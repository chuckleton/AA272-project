%% Generate True Path for Track
% Generates LLA coordinates for track using nominal track dimensions and
% start point LLA from Google Maps.

clc; clear; close all;

latsw = 37.43132258999587;
longsw = -122.16257946186079;
altsw = -11.905701;

latne = 37.43186747665508;
longne = -122.16187352925917;

diff = [latne - latsw; longne - longsw];

latnw = 37.431872;
longnw = -122.163222;
altnw = -11.905701;

lla = [latsw longsw altsw;
    latnw longnw altnw];

p = lla2flat(lla,[latsw longsw],0,altsw);

plot(p(:,2),'.')
axis equal
grid on

x1 = linspace(0,84.39);
y1 = zeros(size(x1));

OD = 92.5;
ID = 36.5*2;
total_width = (OD-ID)/2;
lane_width = total_width/8;

theta2 = linspace(-pi/2,pi/2);
x2 = (36.5 + 4.5*lane_width)*cos(theta2) + 84.39;
y2 = (36.5 + 4.5*lane_width)*sin(theta2) + 36.5 + 4.5*lane_width;

x3 = linspace(x2(end),x2(end)-84.39);
y3 = y2(end)*ones(size(x3));

theta4 = linspace(pi/2,3*pi/2);
x4 = (36.5 + 4.5*lane_width)*cos(theta4);
y4 = (36.5 + 4.5*lane_width)*sin(theta4) + 36.5 + 4.5*lane_width;

plot(x1,y1)
hold on
plot(x2,y2)
plot(x3,y3)
plot(x4,y4)
axis equal

xs = [x1 x2 x3 x4];
ys = [y1 y2 y3 y4];

t = deg2rad(46.2);

Rz = [cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1];

xyzRz = Rz*[xs;-ys;zeros(size(xs))];

plot(xyzRz(1,:),xyzRz(2,:))

lla = flat2lla(xyzRz',[latsw longsw],0,altsw);

figure()
geoplot(lla(:,1),lla(:,2),'-.')
geobasemap satellite

% writematrix(lla,'track_coords.csv')