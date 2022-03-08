function lla = convert_to_lla(xy, initial)

xy = cell2mat(xy(2:end,2:3));
xy = [xy(:,1) -xy(:,2) zeros(size(xy,1),1)];
lla = flat2lla(xy,initial,90,0);

end