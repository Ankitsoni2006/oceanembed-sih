const { mdToPdf } = require('md-to-pdf');

(async () => {
	await mdToPdf(
		{ path: 'docs/SIH26066_TEAM_MASTER_DECISION_GUIDE.md' },
		{
			dest: 'docs/SIH26066_TEAM_MASTER_DECISION_GUIDE.pdf',
			launch_options: { args: ['--no-sandbox', '--disable-setuid-sandbox'] },
			pdf_options: { format: 'A4', margin: '20mm', printBackground: true },
			css: `
				body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #222; }
				h1, h2, h3 { color: #004085; border-bottom: 1px solid #eee; padding-bottom: 5px; }
				table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }
				th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
				th { background-color: #f8f9fa; color: #333; }
				.page-break { page-break-after: always; }
				code { background: #f4f4f4; padding: 2px 5px; border-radius: 4px; }
				pre code { display: block; padding: 10px; overflow-x: auto; font-size: 13px; }
			`
		}
	).catch(console.error);
})();
