### Issues
Slice size: 
- 100 OK but there are too much slices files, hard to manage
- 500 
    - "java.lang.OutOfMemoryError: Java heap space" at joern_create() "importCpg" line (solved see bottom)
        - increase max heap size, see bottom section
        - increase process timeout in cpg_generator.py in joern_create()

In CPG_generator
- Generate json with joern_create: some Cpg.bin has methods that freeze the generation when parsing the PDG edges
    - this happens on few samples (4 for me) 
    - my solution is to skip these methods parsing with graph-for-funcs_DEBUG.sc manually with joern (parse, see debug prints and then change line 77 and re-parse)   

### Dataset preprocess
Look at `select` function in `run.py`
### Joern version v1.0.170
[Joern releases](https://github.com/joernio/joern/releases?expanded=true&page=171&q=v1.0.170)

[Joern version v1.0.170](https://github.com/joernio/joern/releases/tag/v1.0.170)

Download [joern-cli.zip](https://github.com/joernio/joern/releases/download/v1.0.170/joern-cli.zip) and extract it in /joern

#### Increse JVM heap size for joern 
Open the script of joern (joern/joern-cli/joern) and change last line to 
```
$SCRIPT -J-XX:+UseG1GC -J-XX:CompressedClassSpaceSize=128m -Dlog4j.configurationFile="$SCRIPT_ABS_DIR"/conf/log4j2.xml -J-XX:+UseStringDeduplication -J-Xmx12g "$@"
```
Speficly the -Xmx12g define the heap maximum size (2g,4g,8g,12g,16g...), even if you have only 8Gb of RAM you can use higher value (the system will use the swap area).

#### To see the heap usage and maximum capacity of a Java process
```
jstat -gccapacity <pid> 1000 10 
```
1000 10 represent the refresh rate and the number of total output to print

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