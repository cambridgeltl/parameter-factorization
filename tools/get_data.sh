set -e

BASEDIR=$PWD
DATADIR=$PWD/data
TOOLSDIR=$PWD/tools
UDVERSION=ud-treebanks-v2.4

mkdir -p $DATADIR

# Obtain UD Treebanks for POS tagging
wget -O $DATADIR/$UDVERSION.tgz https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/$UDVERSION.tgz?sequence=4&isAllowed=y

tar -xvzf $DATADIR/$UDVERSION.tgz -C $DATADIR
rm $DATADIR/$UDVERSION.tgz
python3 $TOOLSDIR/create_iso_639_3_symlinks.py $DATADIR/$UDVERSION $DATADIR/${UDVERSION}-symlinked | bash

# Obtain Wikiann for NER
wget -r -l1 -H -t1 -nd -N -np -P $DATADIR/wikiann -A .tar.gz -erobots=off https://blender.cs.illinois.edu/wikiann/
find $DATADIR/wikiann -name '*.tar.gz' -execdir tar -xzvf '{}' \;
rm $DATADIR/wikiann/*.tgz
python $TOOLSDIR/split_wikiann.py $DATADIR/wikiann/
rm $DATADIR/wikiann/wikiann-*.bio
