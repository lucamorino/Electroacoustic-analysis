/*
================================================================================
SCMIR analysis of one audio file
================================================================================

Extracts SCMIR descriptors from a target file, summarises them per segment and
looks for sectional boundaries, following

    Nick Collins, "Noise Music Information Retrieval", in Noise in and as Music
    (University of Huddersfield Press, 2013), doi:10.5920/noise.05

with the SCMIR library (https://github.com/sicklincoln/SCMIR).

The defaults reproduce the paper's set-up: six features (perceptual loudness,
transientness, sensory dissonance, spectral crest, spectral centroid, spectral
entropy), summarised as means and maxima over one-second windows, a whole-file
summary, and SCMIR's findSections change points.

RUN IT
  In the SuperCollider IDE: edit SETTINGS below, click anywhere inside the
  block that begins on the "(" line, and press Cmd+Return (macOS) or
  Ctrl+Return. Progress and results appear in the Post window.

  From a terminal:
    sclang scmir_analysis.scd /path/to/piece.wav [/path/to/output/folder]

  Run scmir_check_install.scd first: it checks the installation without
  analysing anything.

README.md in this folder covers installation and what each output means.
================================================================================
*/

(
var settings = (

	// -------------------------------------------------------------- input
	// The audio file to analyse. A path given on the command line replaces it.
	// SCMIR analyses at 44.1 kHz and expects 44.1 or 48 kHz input.
	target: "~/Music/piece.wav",

	// Analyse an excerpt: start and duration in seconds (duration 0 = to the end).
	start: 0,
	duration: 0,

	// ------------------------------------------------------------- output
	// nil = a folder called "scmir" beside the audio file.
	outputDir: nil,

	// -------------------------------------------------------- descriptors
	// Each entry is [\Name] or [\Name, parameters...], as SCMIRAudioFile takes
	// them. The default is the paper's six; the others SCMIR supports are
	// listed at the bottom of this file.
	features: [
		[\Loudness],            // perceptual loudness (sones)
		[\Transient],           // "transientness": wavelet transient energy
		[\SensoryDissonance],   // sensory dissonance; args: maxpeaks, peakthreshold, norm, clamp
		[\FFTCrest],            // spectral crest: how peaky the spectrum is
		[\SpecCentroid],        // spectral centroid (Hz)
		[\SpectralEntropy]      // spectral entropy
	],

	// When a feature's UGens are not installed: drop that feature and carry on
	// (true), or stop (false).
	skipMissingFeatures: true,

	// Analysis hop in samples: 1024 (~23 ms, SCMIR's default) or 512 (~12 ms).
	// SCMIR supports only these two.
	frameHop: 1024,

	// ------------------------------------------------------- segmentation
	// How the piece is sliced before summarising:
	//   \fixed   constant windows (the paper uses one-second windows)
	//   \onsets  from one SCMIR onset to the next
	//   \frames  every analysis frame, unsummarised (~43 rows per second)
	segmentation: \fixed,

	window: 1.0,          // \fixed: window length, seconds
	hop: nil,             // \fixed: step between windows; nil = window (no overlap)

	onsetThreshold: 0.5,  // \onsets: lower finds more onsets
	onsetODF: \power,     // \onsets: \power \magsum \complex \rcomplex \phase \wphase \mkl
	                      //   SCMIR's own default is \rcomplex, which over-triggers on
	                      //   low sustained partials - see README. \power is steadier.
	onsetMinGap: 10,      // \onsets: minimum gap, in 512-sample frames (~0.12 s)
	minSegment: 0.05,     // \onsets: boundaries closer than this (seconds) are merged

	// Statistics per segment: \mean \max \min \stdev \median (SCMIR's
	// gatherFeaturesBySegments). The paper uses \mean and \max.
	stats: [\mean, \max],

	// ----------------------------------------------------------- sections
	// SCMIR findSections: change points from a novelty curve over the
	// self-similarity matrix of the min-max normalised features. Needs SCMIR's
	// helper programs (scmirexec); skipped with a message when they are absent.
	findSections: true,
	sectionMetric: 0,     // 0 cosine (SCMIR's default), 1 Manhattan, 2 Euclidean
	sectionUnit: 10,      // analysis frames per similarity cell (10 x 23 ms = 0.23 s)
	sectionKernel: 43,    // cells in the novelty window (43 x 0.23 s = about 10 s)

	// ------------------------------------------------------------- extras
	// Also write each segment's values min-max normalised within this file
	// (0..1), the scale the paper reports.
	writeNormalised: true,
	// Also write REAPER marker/region CSVs beside the Audacity label tracks.
	writeReaper: false
);

var checkOnly = ~scmirCheckOnly ? false;
var needs, statCodes, odfs, hint, checkInstall, openTarget, width, featureNames;
var fixedSegments, onsetSegments, fmt, quote, toJSON, writeText, writeCSV;
var writeAudacity, writeReaperCSV, writeRegions, analyse, finish, warnings;

~scmirCheckOnly = nil;
warnings = List[];

// Command line: sclang scmir_analysis.scd <audio> [<output folder>]
if(thisProcess.argv.size > 0) { settings[\target] = thisProcess.argv[0] };
if(thisProcess.argv.size > 1) { settings[\outputDir] = thisProcess.argv[1] };


// =============================================================================
// What each SCMIR feature needs installed, and where it comes from
// =============================================================================
needs = (
	Loudness: [\Loudness], MFCC: [\MFCC], SpecCentroid: [\SpecCentroid],
	SpecPcile: [\SpecPcile], SpecFlatness: [\SpecFlatness], Onsets: [\Onsets],
	RMS: [\RunningSum], ZCR: [\ZeroCrossing], Tempo: [\BeatTrack],
	KeyTrack: [\KeyTrack], KeyClarity: [\KeyClarity], KeyMode: [\KeyMode],
	SensoryDissonance: [\SensoryDissonance], SpectralEntropy: [\SpectralEntropy],
	Chromagram: [\Chromagram], AttackSlope: [\AttackSlope],
	OnsetStatistics: [\OnsetStatistics], BeatStatistics: [\BeatStatistics],
	FFTCrest: [\FFTCrest], FFTSpread: [\FFTSpread], FFTSlope: [\FFTSlope],
	Tartini: [\Tartini], Transient: [\DWT, \WT_Transient, \WT_ModulusSum],
	PolyPitch: [\PolyPitch], PianoPitch: [\PianoPitch], CustomFeature: []
);

hint = { |cls|
	case
	{ [\FeatureSave, \SensoryDissonance, \SpectralEntropy, \Chromagram, \AttackSlope,
		\OnsetStatistics, \BeatStatistics, \KeyClarity, \KeyMode].includes(cls) }
	{ "sc3-plugins (SCMIRUGens)" }
	{ [\FFTCrest, \FFTSpread, \FFTSlope].includes(cls) } { "sc3-plugins (MCLDUGens)" }
	{ cls == \Tartini } { "sc3-plugins (PitchDetection)" }
	{ [\DWT, \WT_Transient, \WT_ModulusSum].includes(cls) }
	{ "Nick Collins's wavelet UGens, which are not part of sc3-plugins (see README)" }
	{ [\PolyPitch, \PianoPitch].includes(cls) }
	{ "a separate Nick Collins UGen package, not part of sc3-plugins" }
	{ "SuperCollider itself - check the installation" };
};

statCodes = (mean: 0, max: 1, min: 2, stdev: 3, median: 4);
odfs = [\power, \magsum, \complex, \rcomplex, \phase, \wphase, \mkl];


// =============================================================================
// Installation check
// =============================================================================
// Returns (errors:, usable:). Every SCMIR class is reached through
// \Name.asClass: writing the class name directly would make this whole block
// fail to parse when SCMIR is missing, before it could say so.
checkInstall = {
	var errors = List[], usable = List[], scmir = \SCMIR.asClass, execDir, missing;

	if(\SCMIRAudioFile.asClass.isNil) {
		errors.add("SCMIR is not installed: copy the SCMIRExtensions folder into\n    %\n  then recompile the class library (Language > Recompile Class Library).".format(Platform.userExtensionDir))
	};
	if(\FeatureSave.asClass.isNil) {
		errors.add("FeatureSave is missing. SCMIR needs it for every extraction; it comes with " ++ hint.(\FeatureSave) ++ ".")
	};

	// No backslashes inside .format strings below: format reads "\%" as an
	// escaped literal percent sign, which shifts every argument after it.
	settings[\features].do { |group|
		var name = group[0].asSymbol, required = needs[name], absent, message;
		if(required.isNil) {
			errors.add("unknown feature %: check the spelling against the list at the bottom of this file.".format("\\" ++ name))
		} {
			absent = required.reject { |cls| cls.asClass.notNil };
			if(absent.isEmpty) { usable.add(group) } {
				message = "% needs %, from %".format("\\" ++ name, absent.join(", "), hint.(absent[0]));
				if(settings[\skipMissingFeatures]) {
					warnings.add(message ++ ". Dropped from this run.")
				} { errors.add(message ++ ".") }
			}
		}
	};
	if(usable.isEmpty and: { errors.isEmpty }) {
		errors.add("none of the requested features can be computed with what is installed.")
	};

	if(settings[\findSections] and: { scmir.notNil }) {
		// SCMIR stores this path shell-escaped; File.exists wants it plain.
		execDir = scmir.executabledirectory.replace("\\ ", " ");
		missing = ["similaritymatrix", "noveltycurve"].reject { |name| File.exists(execDir ++ name) };
		if(missing.notEmpty) {
			warnings.add("findSections needs SCMIR's helper programs % in\n    %\n  Sections will be skipped (see README, \"Helper programs\").".format(missing.join(" and "), execDir.standardizePath));
			settings[\findSections] = false;
		}
	};

	if([512, 1024].includes(settings[\frameHop]).not) {
		errors.add("frameHop must be 512 or 1024: SCMIR supports no other hop.")
	};
	if((settings[\segmentation] == \onsets) and: { odfs.includes(settings[\onsetODF]).not }) {
		errors.add("onsetODF must be one of %.".format(odfs))
	};
	settings[\stats].do { |stat|
		if(statCodes[stat.asSymbol].isNil) {
			errors.add("unknown statistic " ++ ("\\" ++ stat) ++ ": use \\mean \\max \\min \\stdev or \\median.")
		}
	};
	if([\fixed, \onsets, \frames].includes(settings[\segmentation]).not) {
		errors.add("segmentation must be \\fixed, \\onsets or \\frames.")
	};

	(errors: errors, usable: usable.asArray)
};

openTarget = { |path|
	var file = SoundFile.openRead(path), info;
	if(file.notNil) {
		info = (path: path, sampleRate: file.sampleRate, numFrames: file.numFrames,
			channels: file.numChannels, length: file.duration,
			header: file.headerFormat, format: file.sampleFormat);
		file.close;
	};
	info
};


// =============================================================================
// Feature names (one column per output; MFCC, Chromagram etc. have several)
// =============================================================================
// Mirrors SCMIRAudioFile:setFeatureInfo, which is what decides numfeatures.
width = { |group|
	switch(group[0].asSymbol,
		\MFCC, { group[1] }, \Chromagram, { group[1] }, \SpectralEntropy, { group[1] ? 1 },
		\Tartini, { if(group.size == 1) { 1 } { 2 } },
		\PolyPitch, { (2 * group[1]) + 1 }, \PianoPitch, { group[1] },
		\OnsetStatistics, { 3 }, \BeatStatistics, { 4 }, \CustomFeature, { group[2] ? 1 },
		{ 1 })
};

featureNames = { |file|
	var names = List[];
	file.featureinfo.do { |group|
		var count = width.(group), name = group[0].asString;
		if(count == 1) { names.add(name) } {
			count.do { |i| names.add(name ++ "_" ++ i.asString.padLeft(2, "0")) }
		}
	};
	if(names.size != file.numfeatures) {
		warnings.add("could not name the % feature outputs individually; columns are f00, f01, ...".format(file.numfeatures));
		names = List.fill(file.numfeatures, { |i| "f" ++ i.asString.padLeft(2, "0") });
	};
	names.asArray
};


// =============================================================================
// Segmentation
// =============================================================================
// Pairs of [start, end] in seconds from the start of the analysed excerpt.
fixedSegments = { |length, window, step, minimum|
	var out = List[], start = 0.0, end;
	if((window <= 0) or: { step <= 0 }) { Error("window and hop must be positive").throw };
	while { start < length } {
		end = (start + window).min(length);
		// A trailing scrap shorter than one analysis frame has no data in it.
		if((end - start) >= minimum) { out.add([start, end]) };
		start = start + step;
	};
	out.asArray
};

onsetSegments = { |onsets, length, minimum|
	var edges = List[0.0];
	onsets.asArray.sort.do { |time|
		if((time > 0) and: { time < length } and: { (time - edges.last) >= minimum }) {
			edges.add(time)
		}
	};
	if((edges.size > 1) and: { (length - edges.last) < minimum }) { edges.pop };
	edges.add(length);
	Array.fill(edges.size - 1, { |i| [edges[i], edges[i + 1]] })
};


// =============================================================================
// Writing
// =============================================================================
fmt = { |value|
	case
	{ value.isNil } { "" }
	{ value.isNaN } { "nan" }
	{ value.isInteger } { value.asString }
	{ value.asStringPrec(10) }
};

quote = { |string|
	"\"" ++ string.replace("\\", "\\\\").replace("\"", "\\\"")
	.replace("\n", "\\n").replace("\t", "\\t") ++ "\""
};

toJSON = { |value, indent = ""|
	var inner = indent ++ "  ";
	case
	{ value.isNil } { "null" }
	{ value == true } { "true" }
	{ value == false } { "false" }
	{ value.isKindOf(SimpleNumber) } {
		if(value.isNaN or: { value.abs == inf }) { "null" } {
			if(value.isInteger) { value.asString } { value.asStringPrec(12) }
		}
	}
	{ value.isKindOf(Symbol) } { quote.(value.asString) }
	{ value.isKindOf(String) } { quote.(value) }
	{ value.isKindOf(Dictionary) } {
		if(value.isEmpty) { "{}" } {
			"{\n" ++ value.keys.asArray.sort { |a, b| a.asString < b.asString }.collect { |key|
				inner ++ quote.(key.asString) ++ ": " ++ toJSON.(value[key], inner)
			}.join(",\n") ++ "\n" ++ indent ++ "}"
		}
	}
	{ value.isKindOf(SequenceableCollection) } {
		if(value.isEmpty) { "[]" } { "[" ++ value.collect { |item| toJSON.(item, inner) }.join(", ") ++ "]" }
	}
	{ quote.(value.asCompileString) }
};

writeText = { |path, text|
	var file = File(path, "w");
	if(file.isOpen.not) { Error("could not write " ++ path).throw };
	file.write(text);
	file.close;
	path
};

writeCSV = { |path, header, rows|
	var file = File(path, "w");
	if(file.isOpen.not) { Error("could not write " ++ path).throw };
	file.write(header.join(",") ++ "\n");
	rows.do { |row| file.write(row.join(",") ++ "\n") };
	file.close;
	path
};

// Audacity: File > Import > Labels...
writeAudacity = { |path, regions|
	writeText.(path, regions.collect { |region|
		fmt.(region[0]) ++ "\t" ++ fmt.(region[1]) ++ "\t" ++ region[2] ++ "\n"
	}.join)
};

// REAPER: Region/Marker Manager > Import... (same layout REAPER exports)
writeReaperCSV = { |path, regions|
	writeText.(path, "#,Name,Start,End,Length,Color\n" ++ regions.collect { |region, i|
		"R" ++ (i + 1) ++ "," ++ region[2] ++ "," ++ fmt.(region[0]) ++ "," ++ fmt.(region[1])
		++ "," ++ fmt.(region[1] - region[0]) ++ ",\n"
	}.join)
};

writeRegions = { |base, regions, written|
	written.add(writeAudacity.(base ++ ".labels.txt", regions));
	if(settings[\writeReaper]) { written.add(writeReaperCSV.(base ++ ".reaper.csv", regions)) };
};


// =============================================================================
// Analysis
// =============================================================================
analyse = { |usable|
	var started = Main.elapsedTime, scmir = \SCMIR.asClass, path, info, file, from, length;
	var fraction, durationArg, offset, names, count, raw, hop, segments, stats, codes;
	var rawStats, normStats, normFrames, summary, sections, metric, wantNorm, column, negative;
	var outDir, base, written = List[], header, rows, columnsFor, regions, edges, table;

	path = settings[\target].standardizePath;
	info = openTarget.(path);
	if(info.isNil) {
		Error("cannot open the audio file\n    %\n  Set `target` in SETTINGS, or pass a path on the command line.".format(path)).throw
	};
	if([44100, 48000].includes(info[\sampleRate].asInteger).not) {
		warnings.add("the file is % Hz. SCMIR analyses at 44.1 kHz and expects 44.1 or 48 kHz input; convert it for reliable spectral features.".format(info[\sampleRate]))
	};

	from = settings[\start].clip(0, info[\length]);
	length = if(settings[\duration] > 0) {
		settings[\duration].min(info[\length] - from)
	} { info[\length] - from };
	if(length <= 0) {
		Error("nothing to analyse: start (% s) is at or past the end of the file (% s).".format(from, info[\length])).throw
	};
	// SCMIRAudioFile takes the start as a FRACTION of the file and the duration
	// in seconds, and reads a duration of 0 as "the whole file" even when a
	// start is given - so always pass an explicit duration for an excerpt.
	fraction = from / info[\length];
	durationArg = if((from > 0) or: { settings[\duration] > 0 }) { length } { 0 };

	"\n[1/5] Reading %".format(path).postln;
	"      % Hz, % channel(s), % s; analysing % s from % s".format(info[\sampleRate],
		info[\channels], info[\length].round(0.01), length.round(0.01), from.round(0.01)).postln;

	scmir.setFrameHop(settings[\frameHop]);
	file = \SCMIRAudioFile.asClass.new(path, usable, 0, fraction, durationArg);
	if(file.valid.not) { Error("SCMIR could not open " ++ path).throw };
	offset = file.loadstart / info[\sampleRate];
	names = featureNames.(file);
	count = file.numfeatures;

	"\n[2/5] Extracting % (% output(s), hop % samples)".format(
		usable.collect { |group| group[0] }, count, settings[\frameHop]).postln;
	file.extractFeatures(false);   // raw values; normalising is done separately below
	if(file.featuredata.isNil or: { file.numframes < 1 }
		or: { file.featuredata.size != (file.numframes * count) }) {
		Error("feature extraction returned no data. The usual cause is a server plugin that is missing even though its class exists - look for \"UGen not installed\" in the messages above.").throw
	};
	raw = file.featuredata.copy;
	hop = scmir.hoptime;
	"      % frames (% per second)".format(file.numframes, scmir.framerate.round(0.01)).postln;

	stats = settings[\stats].collect(_.asSymbol);
	codes = stats.collect { |stat| statCodes[stat] };
	// gatherFeaturesBySegments starts its running max at 0.0, so a feature whose
	// values are all negative within a segment would read 0. Say so if it can happen.
	if(stats.includes(\max)) {
		negative = names.select { |name, k|
			var lowest = inf;
			file.numframes.do { |i| lowest = lowest.min(raw[(i * count) + k]) };
			lowest < 0
		};
		if(negative.notEmpty) {
			warnings.add(negative.join(", ") ++ " can be negative. SCMIR's max summary starts from 0, so a segment whose values are all negative reports a max of 0 for these.")
		}
	};
	// SensoryDissonance clamps at 1.0 by default, and dense noise pins it there -
	// Collins met exactly this with Merzbow. Measure how often it happens.
	column = 0;
	file.featureinfo.do { |group|
		var clamp, pinned = 0, share;
		if(group[0].asSymbol == \SensoryDissonance) {
			clamp = group[4] ? 1.0;
			file.numframes.do { |i|
				if(raw[(i * count) + column] >= (clamp * 0.9999)) { pinned = pinned + 1 }
			};
			share = pinned / file.numframes;
			if(share > 0.01) {
				warnings.add("SensoryDissonance sat at its clamp (" ++ clamp ++ ") in "
					++ (share * 100).round(0.1) ++ "% of frames, so its max, and part of its mean, are clipped."
					++ " Collins met the same with Merzbow. Raise the clamp, the 4th parameter:"
					++ " [\\SensoryDissonance, 100, 0.1, nil, 10]. Values can then exceed 1, so compare only runs that use the same clamp.")
			}
		};
		column = column + width.(group);
	};

	"\n[3/5] Segmenting: %".format(settings[\segmentation]).postln;
	segments = switch(settings[\segmentation],
		\fixed, {
			fixedSegments.(length, settings[\window], settings[\hop] ? settings[\window], hop)
		},
		\onsets, {
			file.extractOnsets(settings[\onsetThreshold], settings[\onsetODF], 1, 0.1,
				settings[\onsetMinGap], 11, 1);
			"      % onsets".format(file.numonsets).postln;
			onsetSegments.(file.onsetdata ? [], length, settings[\minSegment])
		},
		\frames, { nil }
	);
	if(segments.notNil) {
		"      % segments".format(segments.size).postln;
		rawStats = codes.collect { |code| file.gatherFeaturesBySegments(segments, false, code) };
	};
	summary = codes.collect { |code| file.gatherFeaturesBySegments([[0, length]], false, code) };

	wantNorm = settings[\writeNormalised] or: { settings[\findSections] };
	if(wantNorm) {
		// Per-file min-max to 0..1 (SCMIR normalisation type 0), as in the paper.
		file.renormalize;
		if(segments.notNil) {
			normStats = codes.collect { |code| file.gatherFeaturesBySegments(segments, false, code) }
		} { normFrames = file.featuredata.copy };
	};

	"\n[4/5] Sections".postln;
	if(settings[\findSections]) {
		metric = settings[\sectionMetric];
		if((metric == 0) and: { count < 2 }) {
			warnings.add("cosine distance is always 0 between single-feature vectors, so sections use Euclidean distance (2) instead.");
			metric = 2;
		};
		// Runs on the normalised frames; must come before anything replaces
		// featuredata with per-segment data, or it reads segments as frames.
		sections = file.findSections(metric, settings[\sectionUnit], settings[\sectionKernel])
		.asArray.collect { |time| time + offset };
		"      boundaries at % s".format(sections.collect(_.round(0.01))).postln;
	} { "      skipped".postln };

	// ----------------------------------------------------------------- write
	"\n[5/5] Writing".postln;
	outDir = (settings[\outputDir] ?? { PathName(path).pathOnly +/+ "scmir" }).standardizePath;
	File.mkdir(outDir);
	base = PathName(path).fileNameWithoutExtension;
	if((from > 0) or: { settings[\duration] > 0 }) {
		base = base ++ "_" ++ from.round(0.01).asString ++ "-" ++ (from + length).round(0.01).asString ++ "s"
	};
	base = outDir +/+ base;

	header = ["segment", "start", "end", "duration"];
	columnsFor = { |suffixes|
		header ++ names.collect { |name| suffixes.collect { |suffix| "scmir." ++ name ++ "." ++ suffix } }.flatten
	};

	if(segments.notNil) {
		table = { |data|
			segments.collect { |segment, i|
				var row = List[i.asString, fmt.(segment[0] + offset), fmt.(segment[1] + offset),
					fmt.(segment[1] - segment[0])];
				count.do { |k| data.do { |values| row.add(fmt.(values[(i * count) + k])) } };
				row.asArray
			}
		};
		written.add(writeCSV.(base ++ ".scmir.segments.csv", columnsFor.(stats), table.(rawStats)));
		if(settings[\writeNormalised]) {
			written.add(writeCSV.(base ++ ".scmir.segments.normalised.csv", columnsFor.(stats), table.(normStats)))
		};
	} {
		table = { |data|
			Array.fill(file.numframes, { |i|
				var start = i * hop, end = ((i + 1) * hop).min(length), row;
				row = List[i.asString, fmt.(start + offset), fmt.(end + offset), fmt.(end - start)];
				count.do { |k| row.add(fmt.(data[(i * count) + k])) };
				row.asArray
			})
		};
		written.add(writeCSV.(base ++ ".scmir.frames.csv", columnsFor.(["value"]), table.(raw)));
		if(settings[\writeNormalised]) {
			written.add(writeCSV.(base ++ ".scmir.frames.normalised.csv", columnsFor.(["value"]), table.(normFrames)))
		};
	};

	// Whole-file summary: one row, raw values (the paper's "summary features").
	written.add(writeCSV.(base ++ ".scmir.summary.csv",
		["file", "start", "end", "duration", "frames"]
		++ names.collect { |name| stats.collect { |stat| "scmir." ++ name ++ "." ++ stat } }.flatten,
		[[PathName(path).fileNameWithoutExtension, fmt.(offset), fmt.(offset + length), fmt.(length),
			file.numframes.asString]
		++ names.collect { |name, k| summary.collect { |values| fmt.(values[k]) } }.flatten]
	));

	if(settings[\segmentation] == \onsets) {
		writeRegions.(base ++ ".scmir.onsets", segments.collect { |segment, i|
			[segment[0] + offset, segment[1] + offset, i.asString.padLeft(4, "0")]
		}, written);
	};

	if(sections.notNil) {
		edges = [offset] ++ sections ++ [offset + length];
		regions = Array.fill(edges.size - 1, { |i| [edges[i], edges[i + 1], "S" ++ (i + 1)] });
		written.add(writeCSV.(base ++ ".scmir.sections.csv", ["section", "start", "end", "duration"],
			regions.collect { |region| [region[2], fmt.(region[0]), fmt.(region[1]), fmt.(region[1] - region[0])] }));
		writeRegions.(base ++ ".scmir.sections", regions, written);
	};

	written.add(base ++ ".scmir.run.json");
	writeText.(base ++ ".scmir.run.json", toJSON.((
		tool: "Electroacoustic-analysis supercollider/scmir_analysis.scd",
		method: "Collins (2013), Noise Music Information Retrieval, doi:10.5920/noise.05; SCMIR",
		date: Date.getDate.asString,
		supercollider: Main.version,
		audio: (path: path, sampleRate: info[\sampleRate], channels: info[\channels],
			fileDuration: info[\length], start: offset, duration: length),
		settings: settings,
		analysis: (frameHop: settings[\frameHop], framesPerSecond: scmir.framerate,
			frames: file.numframes, features: usable, columns: names,
			segments: segments !? { segments.size }, sectionBoundaries: sections ? []),
		warnings: warnings.asArray,
		outputs: written.asArray,
		elapsedSeconds: (Main.elapsedTime - started).round(0.01)
	)) ++ "\n");

	(file: file, names: names, summary: summary, stats: stats, written: written,
		elapsed: Main.elapsedTime - started)
};


// =============================================================================
// Report
// =============================================================================
finish = { |ok, result|
	var meanAt, maxAt;
	if(warnings.notEmpty) {
		"\nWarnings:".postln;
		warnings.do { |warning| ("  - " ++ warning).postln };
	};
	if(result.notNil) {
		meanAt = result[\stats].indexOf(\mean);
		maxAt = result[\stats].indexOf(\max);
		"\nWhole-file summary (raw values):".postln;
		("  " ++ "feature".padRight(24) ++ (if(meanAt.notNil) { "mean".padLeft(14) } { "" })
			++ (if(maxAt.notNil) { "max".padLeft(14) } { "" })).postln;
		result[\names].do { |name, k|
			("  " ++ name.padRight(24)
				++ (if(meanAt.notNil) { fmt.(result[\summary][meanAt][k]).padLeft(14) } { "" })
				++ (if(maxAt.notNil) { fmt.(result[\summary][maxAt][k]).padLeft(14) } { "" })).postln;
		};
		"\nWrote:".postln;
		result[\written].do { |path| ("  " ++ path).postln };
		"\nDone in % s.".format(result[\elapsed].round(0.1)).postln;
	};
	if(ok.not) { "\nStopped: see the message above.".postln };
	// From a terminal, quit with a status code; in an editor, stay open.
	if(Platform.ideName == "none") { (if(ok) { 0 } { 1 }).exit };
};


// =============================================================================
// Run
// =============================================================================
fork {
	var ok = true, check, result, target;
	"\n=== SCMIR analysis (Collins 2013 method) ===".postln;
	try {
		check = checkInstall.();
		if(check[\errors].notEmpty) {
			"\nCannot run:".postln;
			check[\errors].do { |error| ("  - " ++ error).postln };
			ok = false;
		} {
			if(checkOnly) {
				"\nInstallation OK.".postln;
				"  SuperCollider %".format(Main.version).postln;
				"  Extensions folder: %".format(Platform.userExtensionDir).postln;
				"  Features available: %".format(check[\usable].collect { |group| group[0] }).postln;
				"  Sections (findSections): %".format(if(settings[\findSections]) { "available" } { "unavailable" }).postln;
				target = openTarget.(settings[\target].standardizePath);
				"  Target: %".format(if(target.notNil) {
					"% (% Hz, % ch, % s)".format(settings[\target], target[\sampleRate], target[\channels], target[\length].round(0.01))
				} { "% - not found (set `target` before analysing)".format(settings[\target]) }).postln;
			} {
				result = analyse.(check[\usable]);
			}
		}
	} { |error|
		ok = false;
		("\n" ++ error.errorString).postln;
		// Errors this script raises carry their own explanation; anything else
		// is unexpected, so print the full trace for diagnosis.
		if(error.class != Error) { error.reportError };
	};
	finish.(ok, result);
};
)


/*
================================================================================
FEATURES SCMIR SUPPORTS
================================================================================
Write them in `features` as [\Name] or [\Name, args...]. "core" means built into
SuperCollider; the rest need sc3-plugins unless noted.

  [\Loudness]                  perceptual loudness, sones           core
  [\SpecCentroid]              spectral centroid, Hz                core
  [\SpecPcile, 0.95]           frequency below which 95% of energy  core
  [\SpecFlatness]              spectral flatness                    core
  [\MFCC, 13]                  13 MFCCs                             core
  [\RMS]  [\ZCR]               level, zero-crossing rate            core
  [\Onsets]                    raw onset detection function         core
  [\Tempo]  [\KeyTrack]        beat tempo, key                      core
  [\SensoryDissonance]         sensory dissonance                   SCMIRUGens
  [\SpectralEntropy]           spectral entropy (single band only)  SCMIRUGens
  [\Chromagram, 12]            12 pitch-class energies              SCMIRUGens
  [\AttackSlope]               attack slope                         SCMIRUGens
  [\OnsetStatistics]           onset density statistics (3 values)  SCMIRUGens
  [\BeatStatistics]            beat statistics (4 values)           SCMIRUGens
  [\KeyClarity]  [\KeyMode]    key clarity, mode                    SCMIRUGens
  [\FFTCrest]                  spectral crest                       MCLDUGens
  [\FFTSpread]  [\FFTSlope]    spectral spread, slope               MCLDUGens
  [\Tartini]                   pitch (MIDI); [\Tartini, 1] adds     PitchDetection
                               the pitch confidence
  [\Transient]                 wavelet transientness                Nick Collins's
                                                                    wavelet UGens
  [\PolyPitch, 4]              polyphonic pitch                     separate package
  [\PianoPitch]                piano pitch                          separate package

Parameters, in order (all optional):
  \Loudness           smask (0.25), tmask (1)
  \SensoryDissonance  maxpeaks (100), peakthreshold (0.1), norm, clamp (1.0)
                      Collins raised the clamp for Merzbow, whose dissonance
                      otherwise pinned at the ceiling.
  \FFTCrest           freqlo (0), freqhi (50000)
  \Transient          two WT_Transient thresholds (0.5, 0.1)
================================================================================
*/
