<?php
echo 'ok'
$command= escapeshellcmd('python ./masknew1.py --image https://dockboyz.s3.ap-south-1.amazonaws.com/uploads/agents/Documents/5c7775e1e7191.png');
$output= shell_exec($command);
echo $output;
?>
