Tricks to install horovod successfully with NCCL and openmpi 4
https://github.com/horovod/horovod/issues/1492

In order to resolve similar issue had to:

remove all existing protobuf installs
conda uninstall protobuf
conda uninstall libprotobuf

if you cannot remove it, at least 
1. remove default protoc bin
```
mv protoc protoc_bk
```
2. remove default google protoc header foler
```
mv google google_bk
```

manually clean up /usr/include/google/protobuf/ where /usr/bin/ is the
installation folder of older protobuf
Install latest version of protobuf, not necessarily the exact one used by TF.
sudo rm -rf ./protoc
unzip protoc-3.10.1-linux-x86_64.zip -d protoc
chmod 755 -R protoc
BASE=/usr/local
sudo rm -rf $BASE/include/google/protobuf/
sudo cp protoc/bin/protoc $BASE/bin 
sudo cp -R protoc/include/$BASE/include 


17/06/20
Develop an alternative learning scheme for network parameters and logic
parameters.
