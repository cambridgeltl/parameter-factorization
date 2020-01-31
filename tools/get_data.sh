set -e

BASEDIR=$PWD
DATADIR=$PWD/data
TOOLSDIR=$PWD/tools
UDVERSION=ud-treebanks-v2.4

# Obtain UD Treebanks for POS tagging
wget -O $DATADIR/$UDVERSION.tgz https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/$UDVERSION.tgz?sequence=4&isAllowed=y

tar -xvzf $DATADIR/$UDVERSION.tgz
rm $DATADIR/$UDVERSION.tgz
python3 $TOOLSDIR/create_iso_639_3_symlinks.py $DATADIR/$UDVERSION $DATADIR/${UDVERSION}-symlinked | bash

# Obtain Wikiann for NER
wget -r -l1 -np -nd -P $DATADIR/wikiann "https://blender04.cs.rpi.edu/~panx2/wikiann/data" -A "*.tar.gz"
find $DATADIR/wikiann -name '*.tar.gz' -execdir tar -xzvf '{}' \;
rm $DATADIR/wikiann/*.tgz
python $TOOLSDIR/split_wikiann.py $DATADIR/wikiann/
rm $DATADIR/wikiann/wikiann-*.bio
