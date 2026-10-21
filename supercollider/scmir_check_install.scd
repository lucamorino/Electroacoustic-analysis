/*
Check the SCMIR installation without analysing anything.

Run this once after installing, and again whenever scmir_analysis.scd complains.
It reports, in the Post window: whether SCMIR and the UGens each feature needs
are present, whether section finding is available, where your Extensions
folder is, and whether the `target` set in scmir_analysis.scd can be opened.

It uses the checks inside scmir_analysis.scd, so keep the two files together.

In the IDE: click inside the block below and press Cmd+Return (Ctrl+Return).
From a terminal: sclang scmir_check_install.scd
*/

(
var here = thisProcess.nowExecutingPath;
if(here.isNil) {
	"Save this file next to scmir_analysis.scd, open it from disk, and run it again.".postln;
} {
	~scmirCheckOnly = true;
	(here.dirname +/+ "scmir_analysis.scd").load;
};
)
