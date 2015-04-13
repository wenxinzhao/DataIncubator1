clear;
 load OliverTwist; data1 = data;
load DonQuixote; data2 = data;
load Pride&Prejudice; data3 = data; 
clear data
whos

hf = figure(1); % creates a new figure
set(hf,'Color',[1 1 1],'Name','Basic description of the Dataset'); 

% prints subcollection's names and number of documents 
 subplot(2,2,1); axis off
 for k=1:3 
 collname = eval(sprintf('data%d(1).book',k)); 
 ndocs = eval(sprintf('length(data%d)',k)); 
 comment = sprintf('%s\nDataset %d (%d documents)\n',collname,k,ndocs);
 text(0,1.1-k/3,comment); 
 end 
 
 % plots histogram of document sizes for dataset1 (Oliver Twist) 
 for k=1:length(data1), vals1(k)=sum(data1(k).count); end; 
 [hist1,n1] = hist(vals1,min(vals1):max(vals1));
 figure(5)
 subplot(1,3,1); bar(n1,hist1); axis([61,300,0,25]);
 text(220,23,'Dataset 1'); 
 xlabel('Document Size (in words)'); ylabel('Frequency');
 
 % plots histogram of document sizes for dataset2 (Don Quixote) 
 for k=1:length(data2), vals2(k)=sum(data2(k).count); end; 
 [hist2,n2] = hist(vals2,min(vals2):max(vals2)); 
 subplot(1,3,2); bar(n2,hist2); axis([61,300,0,25]);
 text(220,23,'Dataset 2'); 
 xlabel('Document Size (in words)'); ylabel('Frequency'); 
 
 % plots histogram of document sizes for dataset3 (Pride & Prejudice)
for k=1:length(data3), vals3(k)=sum(data3(k).count); end; 
[hist3,n3] = hist(vals3,min(vals3):max(vals3)); 
subplot(1,3,3); bar(n3,hist3); axis([61,300,0,25]);
text(220,23,'Dataset 3'); 
xlabel('Document Size (in words)'); ylabel('Frequency'); 


% gets the vocabulary of dataset 1 (Oliver Twist)
 vocabulary1 = data1(1).vocab; 
 for k=2:length(data1) 
 vocabulary1 = union(vocabulary1,data1(k).vocab); 
 end 
 
 % gets the vocabulary of dataset 2 (Don Quixote)
 vocabulary2 = data2(1).vocab; 
 for k=2:length(data2) 
 vocabulary2 = union(vocabulary2,data2(k).vocab);
 end 
 
 % gets the vocabulary of dataset 3 (Pride & Prejudice)
 vocabulary3 = data3(1).vocab; 
 for k=2:length(data3) 
 vocabulary3 = union(vocabulary3,data3(k).vocab);
 end
 
 % gets the vocabulary for the overall data collection 
vocabulary = unique([vocabulary1,vocabulary2,vocabulary3]);
 
% generates random indexes for each sub collection
 ridx1 = randperm(length(data1)); 
 ridx2 = randperm(length(data2)); 
 ridx3 = randperm(length(data3)); 
 
 % generates random indexes for test, development and train sets
randomizetst = randperm(300); 
randomizedev = randperm(300);
randomizetrn = randperm(length(data1)+length(data2)+length(data3)-600);

% generates a set or random vectors for initialization purposes
initmtx = rand(60,length(vocabulary));
for k=1:60
initmtx(k,:) = initmtx(k,:)/norm(initmtx(k,:));
end

% extracts the test, development and train datasets
 tst = [data1(ridx1(1:100)),data2(ridx2(1:100)),data3(ridx3(1:100))];
 dev = [data1(ridx1(101:200)),data2(ridx2(101:200)),data3(ridx3(101:200))];
 trn = [data1(ridx1(201:end)),data2(ridx2(201:end)),data3(ridx3(201:end))];

 % randomizes the test, development and train datasets
 tstdata = tst(randomizetst);
 devdata = dev(randomizedev);
 trndata = trn(randomizetrn);
% eliminates unnecessary variables
 clear data1 data2 data3 tst dev trn
 
 data = [tstdata,devdata,trndata];
 
 % initializes the tf matrix and category vector
 ndocs = length(data); nvocs = length(vocabulary);
 tfmtx = sparse(nvocs,ndocs); category = zeros(ndocs,1);
 
 for k = 1:ndocs % updates the tf matrix and category vector
[void,index] = intersect(vocabulary,data(k).vocab);
tfmtx(index,k) = data(k).count;
switch data(k).book
case 'OLIVER TWIST', category(k) = 1;
case 'DON QUIXOTE', category(k) = 2;
case 'PRIDE AND PREJUDICE', category(k) = 3;
end
 end

 % computes the idf vector
idfvt = log(ndocs./full(sum(tfmtx>0,2)));
tfidf = sparse(nvocs,ndocs);
 
 for k=1:ndocs
tfidf(:,k) = tfmtx(:,k).*idfvt;
tfidf(:,k) = tfidf(:,k)/norm(tfidf(:,k));
end;
 
tstmtx = tfidf(:,1:300); tstcat = category(1:300);
devmtx = tfidf(:,301:600); devcat = category(301:600);
trnmtx = tfidf(:,601:end); trncat = category(601:end);

% initializes the parameters for k-means clustering
nclusts = 3; % defines the number of clusters to be considered
dst = 'cosine'; % defines the distance metric to be used
initc = initmtx(1:nclusts,:); % defines the initial set of centroids

% applies the k-means clustering algorithm to the test set
[idxs,centroids] = kmeans(tstmtx',nclusts,'Distance',dst,'Start',initc);

% computes cosine distances among cluster and category centroids
 for k=1:nclusts
for n=1:3
catcentroid = sum(tstmtx(:,tstcat==n)',1)/sum(tstcat==n);
catcentroid = catcentroid/norm(catcentroid);
clucentroid = centroids(k,:)/norm(centroids(k,:));
clu2catdist(n,k) = 1-(catcentroid*clucentroid');
end
 end

% selection of best cluster-to-category assignment
permutations = perms([1,2,3]); % considers all possible assignments
for k=1:size(permutations,1) % computes overall distances for all cases
dist1 = clu2catdist(1,permutations(k,1));
dist2 = clu2catdist(2,permutations(k,2));
dist3 = clu2catdist(3,permutations(k,3));
overalldist(k) = dist1+dist2+dist3;
end
[void,best] = min(overalldist); % gets the best assignment


hf = figure(3);
figtitle = 'Cross-plot between cluster and category indexes';
set(hf,'Color',[1 1 1],'Name',figtitle);
plot(tstcat+randn(size(tstcat))/10,idxs+randn(size(idxs))/10,'.');
xlabel('Actual Category Indexes'); ylabel('Cluster Indexes');

newidxs = zeros(size(idxs));
for k=1:3, newidxs(idxs==permutations(best,k)) = k; 
end

%plot

hf = figure(3);
figtitle = 'Cross-plot between cluster and category indexes';
set(hf,'Color',[1 1 1],'Name',figtitle);
plot(tstcat+randn(size(tstcat))/20,idxs+randn(size(idxs))/20,'.');

xlabel('Actual Category Indexes'); ylabel('Cluster Indexes');

accuracy = sum(newidxs==tstcat)/length(tstcat)*100;

% computes the confusion matrix
 confusion_mtx = zeros(nclusts,3);
 for k=1:nclusts,
     for n=1:3
       confusion_mtx(k,n) = sum((newidxs==k)&(tstcat==n));
     end; 
 end;
 
 % displays the confusion matrix
 string = '             Cat 1 Cat 2 Cat 3';
for k=1:3
  formatted_txt = '%s\nNew Cluster %d:   %2d    %2d   %2d';
  string = sprintf(formatted_txt,string,k,confusion_mtx(k,:));
end;
 disp(string);


