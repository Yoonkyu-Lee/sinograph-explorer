#!/usr/bin/env python
#  -*- coding: utf-8 -*-

from utils import readXmlFile, listSvgFiles
from kanjivg import StrokeGr

def flatlist(iterable_of_lists):
	return sum(iterable_of_lists, [])

def subcomponents(strokes):
	return [
		component
		for component in flatlist(
			[ grp.element ] + subcomponents(grp)
			for grp in strokes.childs
			if isinstance(grp, StrokeGr)
		)
		if component is not None
	]

def kanji_component_dic(kanjiVGs):
	kcdic = {}

	for kanjiVG in kanjiVGs:
		components = set(subcomponents(kanjiVG.strokes))

		# We want the kanji as its own component if it is a radical
		if kanjiVG.strokes.radical == "general":
			components |= { kanjiVG.strokes.element }

		# Add components of all variants of the kanji into its entry
		if len(components) > 0:
			kanji = kanjiVG.strokes.element
			kcdic[kanji] = kcdic.get(kanji, set()) | components

	return kcdic

def kradfile(kanjiVGs):
	return (
		kanji + " : " + ' '.join(sorted(components))
		for kanji, components in kanji_component_dic(kanjiVGs).items()
	)

if __name__ == "__main__":
	# Pick your source. The XML doesn't have the variants.
	#kanjiVGs = readXmlFile('./kanjivg.xml').values()
	kanjiVGs = [ f.read() for f in listSvgFiles("./kanji/") ]

	with open("./kradfile.txt", mode="w") as output_file:
		for line in kradfile(kanjiVGs):
			output_file.write(line+"\n")
