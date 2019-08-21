<?php
echo ‘ok’;
$command= escapeshellcmd('python ./probablyworking.py --image ./reference12.jpg');
$output= shell_exec($command);
echo $output;
?>
this code connects the file to the server
to run the code, u must include #!/bin/bash/python at the start in case youre using python
Also, do not include () in the file name
