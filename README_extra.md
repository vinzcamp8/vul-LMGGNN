### Dataset preprocess
Look at `select` function in `run.py`
### Joern version v1.0.170
[Joern releases](https://github.com/joernio/joern/releases?expanded=true&page=171&q=v1.0.170)

[Joern version v1.0.170](https://github.com/joernio/joern/releases/tag/v1.0.170)

Download [joern-cli.zip](https://github.com/joernio/joern/releases/download/v1.0.170/joern-cli.zip) and extract it in /joern


### Java JDK version 14
14 or previous version as well

Get fresh link from: https://jdk.java.net/14/

Download binary:

```
wget https://download.java.net/java/GA/jdk14.0.2/205943a0976c4ed48cb16f1043c5c647/12/GPL/openjdk-14.0.2_linux-x64_bin.tar.gz
```

Unpack it:
```
tar xvf openjdk-14.0.2_linux-x64_bin.tar.gz
```
Move to jvm folder:
```
mv jdk-14.0.2 /usr/lib/jvm
```
Update java and javac alternatives:
```
update-alternatives --install "/usr/bin/javac" "javac" "/usr/lib/jvm/jdk-14.0.2/bin/javac" 3
update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jdk-14.0.2/bin/java" 3
update-alternatives --set "javac" "/usr/lib/jvm/jdk-14.0.2/bin/javac"
update-alternatives --set "java" "/usr/lib/jvm/jdk-14.0.2/bin/java"
```
Use to switch between versions:
```
update-alternatives --config java
```

### For IVDetect
Install dgl 2.3.0
Pytorch 2.3.0
Torch-sparse 2.3.0
```
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
```
where ${CUDA} should be replaced by either `cpu`, `cu118`, or `cu121` depending on your PyTorch installation.