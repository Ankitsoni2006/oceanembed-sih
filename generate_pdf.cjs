const { mdToPdf } = require('md-to-pdf');

(async () => {
	await mdToPdf(
		{ path: 'OCEANEMBED_DUE_DILIGENCE.md' },
		{ dest: 'OCEANEMBED_DUE_DILIGENCE.pdf', launch_options: { args: ['--no-sandbox', '--disable-setuid-sandbox'] } }
	).catch(console.error);
})();
