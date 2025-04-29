for instance in `/bin/ls -d results/*/*`; do
    echo $instance
    make run command=${instance}/command
done
