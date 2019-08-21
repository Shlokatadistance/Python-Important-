<?php
echo ‘ok’;
$command= escapeshellcmd('python ./Camscanner.py --image ./reference12.jpg');
$output= shell_exec($command);
echo $output;
?>