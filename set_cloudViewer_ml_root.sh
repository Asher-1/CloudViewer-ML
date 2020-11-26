# Sets the env var CLOUDVIEWER_ML_ROOT to the directory of this file.
# The cloudViewer package will use this var to integrate ml3d into a common namespace.
export CLOUDVIEWER_ML_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ $0 == $BASH_SOURCE ]]; then 
        echo "source this script to set the CLOUDVIEWER_ML_ROOT env var."     
else
        echo "CLOUDVIEWER_ML_ROOT is now $CLOUDVIEWER_ML_ROOT"
fi
